import math
import copy
import json
import logging
import time
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, AdamW

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

from transformers import get_wsd_schedule

import wandb

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

from visualization_utils import MinecraftVisualizerPyVista, save_chunks
from data_utils import BlockBiomeConverter, rotate_voxels_90_fix_stairs_torch
import os

# Modified version of lucidrains denoising-diffusion-pytorch implementation: https://github.com/lucidrains/denoising-diffusion-pytorch
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

logger = logging.getLogger(__name__)

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()
        
        raise RuntimeError("This Unet class is 2D-only. For 3D diffusion, please import UNet from 'unet_3d.py'.")

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# Subclass to handle 3D diffusion, since original method is hardcoded to expect images
class GaussianDiffusion3D(GaussianDiffusion):
    def __init__(self, model, *, image_size, auto_normalize=False, **kwargs):
        super().__init__(model, image_size=image_size, auto_normalize=auto_normalize, **kwargs)

        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 3, 'image_size must be an integer or a tuple/list of three integers for 3D data'
        self.image_size = image_size

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, d, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1 1')


        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)


        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        d, h, w = self.image_size
        channels = self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, d, h, w), return_all_timesteps=return_all_timesteps)

    def forward(self, vol, *args, **kwargs):
        b, c, d, h, w, device, vol_size, = *vol.shape, vol.device, self.image_size
        assert d == vol_size[0] and h == vol_size[1] and w == vol_size[2], f'depth, height and width of volume must be {vol_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        vol = self.normalize(vol)
        return self.p_losses(vol, t, *args, **kwargs)


class GaussianDiffusion3D_CFG(GaussianDiffusion3D):
    def __init__(self, model, *, image_size, auto_normalize=False, **kwargs):
        super().__init__(model, image_size=image_size, auto_normalize=auto_normalize, **kwargs)
        self.has_class_conditioning = hasattr(model, 'forward_with_cond_scale')
        self.use_cfg_plus_plus = False

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False, clip_range = (-1., 1.)):
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = clip_range[0], max = clip_range[1]) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = False, clip_range = (-3., 3.)):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi, clip_x_start=clip_denoised, clip_range=clip_range)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(clip_range[0], clip_range[1])

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_losses(self, x_start, t, classes=None, noise=None, offset_noise_strength=None):
        b, c, d, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1 1')

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # self-conditioning logic (if applicable)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, classes).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, classes, x_self_cond) if self.self_condition else self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, clip_range = (-3., 3.)):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised, clip_range = clip_range)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_range = (-3., 3.)):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi, clip_range=clip_range)

        img = self.unnormalize(img)
        return img

    def _sliding_window_starts(self, total, window, step):
        if window > total:
            raise ValueError(f'window size {window} cannot exceed target size {total}')
        starts = list(range(0, total - window + 1, step))
        last_start = total - window
        if len(starts) == 0 or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @torch.no_grad()
    def _windowed_model_predictions(self, img, t, classes, window_size, step, cond_scale, rescaled_phi, clip_denoised = False, clip_range = (-3., 3.)):
        _, _, depth, height, width = img.shape
        wd, wh, ww = window_size

        d_starts = self._sliding_window_starts(depth, wd, step)
        h_starts = self._sliding_window_starts(height, wh, step)
        w_starts = self._sliding_window_starts(width, ww, step)

        pred_noise_acc = torch.zeros_like(img)
        x_start_acc = torch.zeros_like(img)
        counts = torch.zeros((1, 1, depth, height, width), device = img.device, dtype = img.dtype)

        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    img_part = img[:, :, d0:d0+wd, h0:h0+wh, w0:w0+ww]
                    preds_part = self.model_predictions(
                        img_part,
                        t,
                        classes,
                        cond_scale = cond_scale,
                        rescaled_phi = rescaled_phi,
                        clip_x_start = clip_denoised,
                        clip_range = clip_range
                    )

                    pred_noise_acc[:, :, d0:d0+wd, h0:h0+wh, w0:w0+ww] += preds_part.pred_noise
                    x_start_acc[:, :, d0:d0+wd, h0:h0+wh, w0:w0+ww] += preds_part.pred_x_start
                    counts[:, :, d0:d0+wd, h0:h0+wh, w0:w0+ww] += 1

        counts = counts.clamp_min(1.0)
        return ModelPrediction(pred_noise_acc / counts, x_start_acc / counts)

    @torch.no_grad()
    def p_sample_loop_superres(self, classes, shape, window_size, step = 1, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, clip_range = (-3., 3.)):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device = device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'superres sampling loop time step', total = self.num_timesteps):
            batched_times = torch.full((batch,), t, device = device, dtype = torch.long)
            preds = self._windowed_model_predictions(
                img,
                batched_times,
                classes,
                window_size = window_size,
                step = step,
                cond_scale = cond_scale,
                rescaled_phi = rescaled_phi,
                clip_denoised = clip_denoised,
                clip_range = clip_range
            )
            model_mean, _, model_log_variance = self.q_posterior(x_start = preds.pred_x_start, x_t = img, t = batched_times)
            noise = torch.randn_like(img) if t > 0 else 0.
            img = model_mean + (0.5 * model_log_variance).exp() * noise

        return self.unnormalize(img)


    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, clip_range = (-3., 3.)):
        print(f"DDIM sampling")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            preds = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised, clip_range = clip_range)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample_superres(self, classes, shape, window_size, step = 1, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, clip_range = (-3., 3.)):
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)

        for time, time_next in tqdm(time_pairs, desc = 'superres ddim sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            preds = self._windowed_model_predictions(
                img,
                time_cond,
                classes,
                window_size = window_size,
                step = step,
                cond_scale = cond_scale,
                rescaled_phi = rescaled_phi,
                clip_denoised = clip_denoised,
                clip_range = clip_range
            )
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return self.unnormalize(img)


    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7, clip_range = (-3., 3.)):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # Create 3D shape: (batch_size, channels, depth, height, width)
        shape = (batch_size, channels, *image_size)
        return sample_fn(classes, shape, cond_scale, rescaled_phi, clip_range=clip_range)

    @torch.no_grad()
    def sample_superres(self, classes, target_size, window_size = None, step = 1, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = False, clip_range = (-3., 3.)):
        if isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        if window_size is None:
            window_size = self.image_size
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)

        batch_size, channels = classes.shape[0], self.channels
        shape = (batch_size, channels, *target_size)
        sample_fn = self.p_sample_loop_superres if not self.is_ddim_sampling else self.ddim_sample_superres
        return sample_fn(
            classes,
            shape,
            window_size = window_size,
            step = step,
            cond_scale = cond_scale,
            rescaled_phi = rescaled_phi,
            clip_denoised = clip_denoised,
            clip_range = clip_range
        )

    def forward(self, vol, classes=None, *args, **kwargs):
        b, c, d, h, w, device, vol_size = *vol.shape, vol.device, self.image_size
        assert d == vol_size[0] and h == vol_size[1] and w == vol_size[2], f'depth, height and width of volume must be {vol_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        vol = self.normalize(vol)
        return self.p_losses(vol, t, classes, *args, **kwargs)


class DDPMoxelDatasetMemmapConditional(torch.utils.data.Dataset):
    """
    DDPM dataset adapter for *existing* discrete (MD4) processed datasets on disk.

    Input:
      - dir_path/manifest.json
      - dir_path/voxels.npy: integer indices [N, X, Y, Z] (these are "block indices" from block_to_index)
      - dir_path/biome_labels.npy: labels (indices or one-hot)

    Output:
      - voxel embeddings as float tensor [E, Z, X, Y] (DiT/Conv3D layout per-sample: [C, D, H, W])
      - class_label as long scalar tensor

    Conversion path:
      indices -> block_id (via mappings_file_path index_to_block) -> embedding (via block_embeddings_path)
    """

    def __init__(
        self,
        dir_path,
        mappings_file_path: str,
        *,
        rotation_aug_prob: float = 0.0,
        block_embeddings_path: str = "assets/block_embeddings_norm.npy",
        embeddings_dict_key: str | None = None,
        scan_unique_blocks: bool = True,
        scan_target_mb: int = 64,
        crop: bool = False,
        crop_size: int = 16,
    ):
        super().__init__()
        self.crop = crop
        self.crop_size = crop_size
        self.dir_path = str(dir_path)
        self.mappings_file_path = str(mappings_file_path) if mappings_file_path is not None else None
        self.rotation_aug_prob = float(rotation_aug_prob)
        self.block_embeddings_path = str(block_embeddings_path)
        self.embeddings_dict_key = embeddings_dict_key
        self.scan_unique_blocks = bool(scan_unique_blocks)
        self.scan_target_mb = int(scan_target_mb)

        if not self.mappings_file_path or not os.path.exists(self.mappings_file_path):
            raise FileNotFoundError(
                f"mappings_file_path is required and must exist to map indices->block_id. Got: {self.mappings_file_path}"
            )

        manifest_path = os.path.join(self.dir_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.json not found in {self.dir_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        voxels_rel = manifest["paths"]["voxels"]
        labels_rel = manifest["paths"]["biome_labels"]
        self.num_blocks = int(manifest.get("num_blocks", 0) or 0)
        self.num_classes = int(manifest.get("num_classes", 0) or 0)
        self.labels_format = manifest.get("class_labels_format", "indices")

        vox_path = os.path.join(self.dir_path, voxels_rel)
        lbl_path = os.path.join(self.dir_path, labels_rel)

        # Load as numpy memmaps
        self.voxels_np = np.load(str(vox_path), mmap_mode="r")
        self.biomes_np = np.load(str(lbl_path), mmap_mode="r")
        self._is_label_one_hot = (self.biomes_np.ndim == 2)

        # Load mappings: index -> block_id
        mappings = torch.load(self.mappings_file_path, map_location="cpu", weights_only=False)
        block_mappings = mappings.get("block_mappings", {})
        idx_to_block = block_mappings.get("index_to_block", None)
        if not isinstance(idx_to_block, dict) or len(idx_to_block) == 0:
            raise ValueError(
                f"mappings_file_path missing block_mappings.index_to_block dict: {self.mappings_file_path}"
            )
        keys = [int(k) for k in idx_to_block.keys()]
        K = max(keys) + 1
        index_to_block = torch.zeros((K,), dtype=torch.long)
        for k, v in idx_to_block.items():
            index_to_block[int(k)] = int(v)
        self.index_to_block = index_to_block  # [K]
        self.K = int(self.index_to_block.shape[0])
        if self.num_blocks not in (None, 0) and int(self.num_blocks) != int(K):
            # Not fatal, but good to know.
            print(f"[DDPMoxelDatasetMemmapConditional] warning: manifest num_blocks={self.num_blocks} but mappings K={K}")

        # Optional: scan dataset to find which block indices actually appear.
        # This avoids decoding against block IDs that never show up in data.
        self.present_block_indices = None
        self.present_block_ids = None
        if self.scan_unique_blocks:
            uniq = self._scan_unique_indices()
            if uniq:
                bad = [i for i in uniq if (i < 0 or i >= self.K)]
                if bad:
                    show = bad[:10]
                    print(
                        "[DDPMoxelDatasetMemmapConditional] warning: "
                        f"unique index scan found out-of-range ids (K={self.K}). "
                        f"Examples: {show}"
                    )
                uniq_in_range = [i for i in uniq if (0 <= i < self.K)]
                self.present_block_indices = uniq_in_range
                idx_tensor = torch.tensor(uniq_in_range, dtype=torch.long)
                block_ids = self.index_to_block[idx_tensor].unique().tolist()
                self.present_block_ids = sorted({int(b) for b in block_ids})
                print(
                    "[DDPMoxelDatasetMemmapConditional] unique block scan: "
                    f"indices={len(self.present_block_indices)}, "
                    f"block_ids={len(self.present_block_ids)}"
                )

        # Load block_id -> embedding mapping
        self._emb_table = None  # torch.FloatTensor [M, E] (row = block_id)
        # We will attempt to build a dense table for GPU usage.
        
        emb_obj = None
        if self.block_embeddings_path.endswith(".npy"):
            emb_np = np.load(self.block_embeddings_path)
            self._emb_table = torch.from_numpy(np.asarray(emb_np)).float()
        else:
            emb_obj = torch.load(self.block_embeddings_path, map_location="cpu", weights_only=False)
            if self.embeddings_dict_key is not None and isinstance(emb_obj, dict) and self.embeddings_dict_key in emb_obj:
                emb_obj = emb_obj[self.embeddings_dict_key]
            
            if isinstance(emb_obj, dict):
                # Convert dict to dense table
                # Find max block ID
                keys = [int(k) for k in emb_obj.keys()]
                if len(keys) == 0:
                     raise ValueError(f"Empty embeddings dict in {self.block_embeddings_path}")
                M = max(keys) + 1
                
                # Infer E
                first = next(iter(emb_obj.values()))
                if hasattr(first, 'numel'):
                     E = int(first.numel())
                else:
                     E = int(np.prod(first.shape))
                
                # Create table
                # Initialize with zeros or specific strategy? Zeros is safe for unused IDs.
                self._emb_table = torch.zeros((M, E), dtype=torch.float32)
                
                for k, v in emb_obj.items():
                    tv = torch.as_tensor(v).float().view(-1)
                    if tv.numel() != E:
                         raise ValueError(f"Embedding dim mismatch for key {k}: expected {E}, got {tv.numel()}")
                    self._emb_table[int(k)] = tv
            else:
                 # Assume it's a tensor-like object
                 try:
                     self._emb_table = torch.as_tensor(emb_obj).float()
                 except Exception:
                     raise ValueError(
                        f"Could not load embeddings as tensor or dict from {self.block_embeddings_path}, got {type(emb_obj)}"
                    )

        # Validate table
        if self._emb_table is None or self._emb_table.dim() != 2:
             raise ValueError(f"Failed to load valid embedding table [M, E] from {self.block_embeddings_path}")

        self.E = int(self._emb_table.shape[1])
        self.num_embeddings = int(self._emb_table.shape[0])
        self._warned_out_of_range = False

        
        print(
            f"DDPM memmap dataset loaded: voxels={self.voxels_np.shape}, labels={self.biomes_np.shape}, "
            f"K={self.K}, Num Embeddings={self.num_embeddings}, E={self.E}"
        )

    def _scan_unique_indices(self):
        """
        Scan the memmap to find unique block indices present in the dataset.
        This avoids decoding against block IDs that never appear.
        """
        try:
            target_bytes = max(1, int(self.scan_target_mb)) * 1024 * 1024
            sample_bytes = int(self.voxels_np[0].nbytes) if self.voxels_np.shape[0] > 0 else 0
            chunk_n = max(1, int(target_bytes // max(1, sample_bytes)))
        except Exception:
            chunk_n = 16

        uniq = set()
        total = int(self.voxels_np.shape[0])
        for start in range(0, total, chunk_n):
            end = min(start + chunk_n, total)
            batch = self.voxels_np[start:end]
            try:
                uniq.update(np.unique(batch).astype(int).tolist())
            except Exception:
                for v in np.unique(batch):
                    uniq.add(int(v))
        return sorted(uniq)

    def __len__(self):
        return int(self.voxels_np.shape[0])

    def __getitem__(self, index):
        # Voxels are block-indices [X,Y,Z] (np.int32). Convert to torch long.
        idx_np = self.voxels_np[index]
        # Prefer from_numpy to avoid an extra copy; np.asarray handles memmap scalars safely.
        idx = torch.from_numpy(np.asarray(idx_np)).long()  # [X,Y,Z]

        if self.crop:
            s = self.crop_size
            d, h, w = idx.shape
            c_d = (d - s) // 2
            c_h = (h - s) // 2
            c_w = (w - s) // 2
            idx = idx[c_d : c_d + s, c_h : c_h + s, c_w : c_w + s]

        if not self._warned_out_of_range:
            out_of_range = (idx < 0) | (idx >= self.K)
            if out_of_range.any():
                bad_vals = idx[out_of_range].flatten()
                uniq = torch.unique(bad_vals)
                sample = uniq[:10].tolist()
                total = int(out_of_range.sum().item())
                min_val = int(idx.min().item())
                max_val = int(idx.max().item())
                print(
                    "[DDPMoxelDatasetMemmapConditional] warning: "
                    f"found {total} out-of-range indices in a sample "
                    f"(min={min_val}, max={max_val}, K={self.K}). "
                    f"Examples: {sample}"
                )
                self._warned_out_of_range = True

        # Labels: either indices or one-hot
        if self._is_label_one_hot or (self.labels_format == "one_hot"):
            lbl_np = self.biomes_np[index]
            lbl = torch.from_numpy(np.asarray(lbl_np)).long()
            class_label = torch.argmax(lbl).long()
        else:
            class_label = torch.tensor(int(self.biomes_np[index]), dtype=torch.long)

        # Map indices -> block IDs
        # This is fast on CPU (integer lookup)
        idx_safe = idx.clamp(min=0, max=self.K - 1)
        block_ids = self.index_to_block[idx_safe]

        # Optional augmentation: rotation
        if self.rotation_aug_prob > 0.0 and torch.rand(1).item() < self.rotation_aug_prob:
            k = int(torch.randint(1, 4, (1,)).item())
            block_ids = rotate_voxels_90_fix_stairs_torch(block_ids, k=k)

        # Return block_ids (integers) to be embedded on GPU
        # Shape: [32, 32, 32]
        return block_ids, class_label

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        save_only_last_checkpoint = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        self.save_best_and_latest_only = save_best_and_latest_only
        self.save_only_last_checkpoint = save_only_last_checkpoint

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def plot_losses(self):
        """Plot and save the training loss curve"""
        if not self.accelerator.is_main_process:
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add a smoothed version if we have enough data points
        if len(self.losses) > 50:
            import numpy as np
            # Simple moving average
            window_size = min(50, len(self.losses) // 10)
            smoothed_losses = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = self.steps[window_size-1:]
            plt.plot(smoothed_steps, smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the raw loss data
        loss_data = {
            'steps': self.steps,
            'losses': self.losses
        }
        torch.save(loss_data, str(self.results_folder / 'training_loss_data.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        visualizer = MinecraftVisualizerPyVista()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if not self.save_only_last_checkpoint:
                            if self.save_best_and_latest_only:
                                if self.best_fid > fid_score:
                                    self.best_fid = fid_score
                                    self.save("best")
                                self.save("latest")
                            else:
                                self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        if self.accelerator.is_main_process and self.save_only_last_checkpoint:
            self.save("final")

class VoxelTrainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        evaluate_ema_model = True,
        mappings_file_path=None,
        run_name = 'unconditional_unet',
        save_only_last_checkpoint = False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_only_last_checkpoint = save_only_last_checkpoint

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.mappings_file_path = mappings_file_path
        self.run_name = run_name

        self.max_grad_norm = max_grad_norm

        # Loss tracking
        self.losses = []
        self.steps = []

        # dataset and dataloader
        self.ds = dataset
        assert len(self.ds) > 0, 'your dataset is empty'


        num_workers = 8
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        self.evaluate_ema_model = evaluate_ema_model

        self.results_folder = Path(os.path.join(results_folder, self.run_name))
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def plot_losses(self):
        """Plot and save the training loss curve"""
        if not self.accelerator.is_main_process:
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add a smoothed version if we have enough data points
        if len(self.losses) > 50:
            import numpy as np
            # Simple moving average
            window_size = min(50, len(self.losses) // 10)
            smoothed_losses = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = self.steps[window_size-1:]
            plt.plot(smoothed_steps, smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the raw loss data
        loss_data = {
            'steps': self.steps,
            'losses': self.losses
        }
        torch.save(loss_data, str(self.results_folder / 'training_loss_data.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        mappings = torch.load(self.mappings_file_path, weights_only=False)
        converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    # Support datasets that return (voxels, labels) for parity with MD4.
                    # For unconditional DDPM training we ignore labels.
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch
                    data = data.to(device)
                    if self.step == 0:
                        # Converter expects embeddings laid out as [B, E, H, W, D].
                        # Training tensors for Conv3D/DiT are typically [B, E, D, H, W].
                        data_cpu = data.to('cpu')
                        if data_cpu.dim() == 5:
                            # [B,E,D,H,W] -> [B,E,H,W,D]
                            data_cpu = data_cpu.permute(0, 1, 3, 4, 2).contiguous()
                        converted = converter.convert_emb_to_blocks(data_cpu)
                        print(f"converted shape: {converted.shape}")
                        save_chunks(converted, str(self.step) + "_raw_batch", self.results_folder, converter, textured=True)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # Record loss for plotting
                if accelerator.is_main_process:
                    self.losses.append(total_loss)
                    self.steps.append(self.step)

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            if self.evaluate_ema_model:
                                model_to_eval = self.ema.ema_model
                            else:
                                model_to_eval = self.model
                            model_to_eval.eval()
                            all_volumes_list = list(map(lambda n: model_to_eval.sample(batch_size=n), batches))
                            model_to_eval.train()

                        all_volumes = torch.cat(all_volumes_list, dim = 0)

                        # Save the full 3D volume for later inspection
                        torch.save(all_volumes, str(self.results_folder / f'sample-{milestone}.pt'))
                        
                        vols_cpu = all_volumes.to('cpu')
                        if vols_cpu.dim() == 5:
                            vols_cpu = vols_cpu.permute(0, 1, 3, 4, 2).contiguous()
                        converted_samples = converter.convert_emb_to_blocks(vols_cpu)
                        save_chunks(converted_samples, self.step, self.results_folder, converter, textured=True)

                        if not self.save_only_last_checkpoint:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        if self.accelerator.is_main_process and self.save_only_last_checkpoint:
            self.save("final")
        
        # Generate final loss plot
        self.plot_losses()


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class VoxelTrainerClassConditional:
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        optimizer = 'adam',
        weight_decay = 0.0,
        adam_betas = (0.9, 0.99),
        warmup_steps = 0,
        scheduler = 'constant',
        min_lr_ratio = 0.0,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        evaluate_ema_model = True,
        mappings_file_path=None,
        run_name = 'unconditional_unet',
        num_classes = 11,
        default_cond_scale = None,
        dataloader_num_workers: int = 2,
        dataloader_prefetch_factor: int | None = 2,
        dataloader_persistent_workers: bool = True,
        val_dataset = None,
        val_dataloader = None,
        val_batch_size = None,
        val_every_n_steps = None,
        val_max_batches = None,
        val_steps = None,
        val_progress = True,
        compile_model = False,
        use_wandb = False,
        wandb_log_images = True,
        save_only_last_checkpoint = False,
    ):
        super().__init__()

        if compile_model:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",
                mode="default",
                dynamic=False,
            )
        else:
            dynamo_plugin = None

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            step_scheduler_with_optimizer=False,
            dynamo_plugin=dynamo_plugin,
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.num_classes = int(num_classes)
        self.default_cond_scale = None if default_cond_scale in (None, 0) else float(default_cond_scale)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_only_last_checkpoint = save_only_last_checkpoint

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.mappings_file_path = mappings_file_path
        self.run_name = run_name

        self.max_grad_norm = max_grad_norm

        self.losses = []
        self.steps = []
        self.val_losses = []
        self.val_steps = []
        self.best_val_loss = float('inf')
        self.best_val_step = None

        # dataset and dataloader
        self.ds = dataset
        assert len(self.ds) > 0, 'your dataset is empty'

        num_workers = int(dataloader_num_workers)
        prefetch_factor = None if (dataloader_prefetch_factor in (None, 0)) else int(dataloader_prefetch_factor)
        persistent_workers = bool(dataloader_persistent_workers) and num_workers > 0
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        # validation
        self.val_max_batches = None if val_max_batches in (None, 0) else int(val_max_batches)
        _val_loader = None
        if val_dataset is not None:
            _val_loader = DataLoader(
                val_dataset,
                batch_size=(val_batch_size or train_batch_size),
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        elif val_dataloader is not None:
            _val_loader = val_dataloader
        self.val_dl = None if _val_loader is None else self.accelerator.prepare(_val_loader)

        if val_steps not in (None, 0):
            self.val_interval_steps = int(val_steps)
        elif val_every_n_steps not in (None, 0):
            self.val_interval_steps = int(val_every_n_steps)
        else:
            self.val_interval_steps = int(save_and_sample_every)
        self.val_progress = bool(val_progress)

        # optimizer
        opt_name = str(optimizer).lower()
        if opt_name == 'adamw':
            self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=weight_decay)
        elif opt_name == 'adam':
            self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        # scheduler (same as MD4 trainer)
        self.scheduler = get_wsd_schedule(
            optimizer=self.opt,
            num_warmup_steps=int(warmup_steps),
            num_decay_steps=int(train_num_steps) - int(warmup_steps),
            num_training_steps=int(train_num_steps),
            min_lr_ratio=float(min_lr_ratio),
            decay_type=str(scheduler).lower(),
        )

        self.evaluate_ema_model = bool(evaluate_ema_model)
        self.use_wandb = bool(use_wandb)
        self.wandb_log_images = bool(wandb_log_images)

        self.results_folder = Path(os.path.join(results_folder, self.run_name))
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, optimizer, scheduler, dataloader with accelerator
        self.model, self.opt, self.scheduler, dl = self.accelerator.prepare(self.model, self.opt, self.scheduler, dl)
        self.dl = cycle(dl)

        # Cache embedding table on device if possible
        if hasattr(self.ds, '_emb_table') and self.ds._emb_table is not None:
             self.device_emb_table = self.ds._emb_table.to(self.accelerator.device)
        else:
             self.device_emb_table = None

        # EMA (mirror MD4 trainer: update on main only, but keep ema_model on all ranks for eval)
        base_model = self.accelerator.unwrap_model(
            self.model,
            keep_torch_compile=True,
        )
        self.ema = EMA(base_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.ema_model.to(self.accelerator.device)
        self.ema.ema_model.eval().requires_grad_(False)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'best_val_loss': float(self.best_val_loss) if exists(self.best_val_loss) else None,
            'best_val_step': int(self.best_val_step) if exists(self.best_val_step) and self.best_val_step is not None else None,
            'version': __version__
        }
        # Persist compact decode lookup for inference scripts.
        if getattr(self, "_nn_embedding_matrix", None) is not None and getattr(self, "_nn_block_ids", None) is not None:
            data['decode_embedding_matrix'] = self._nn_embedding_matrix.detach().cpu()
            data['decode_block_ids'] = self._nn_block_ids.detach().cpu()

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def _save_best_checkpoint(self):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'best_val_loss': float(self.best_val_loss) if exists(self.best_val_loss) else None,
            'best_val_step': int(self.best_val_step) if exists(self.best_val_step) and self.best_val_step is not None else None,
            'version': __version__
        }
        if getattr(self, "_nn_embedding_matrix", None) is not None and getattr(self, "_nn_block_ids", None) is not None:
            data['decode_embedding_matrix'] = self._nn_embedding_matrix.detach().cpu()
            data['decode_block_ids'] = self._nn_block_ids.detach().cpu()
        torch.save(data, str(self.results_folder / 'model_best.pt'))

    def plot_losses(self):
        """Plot and save training + validation loss curves (MD4-style)."""
        if not self.accelerator.is_main_process:
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.8, label='Train')
        if len(self.val_losses) > 0:
            plt.plot(self.val_steps, self.val_losses, 'orange', linewidth=1.5, alpha=0.9, label='Validation')
            try:
                import numpy as np
                best_idx = int(np.argmin(self.val_losses))
                best_step = self.val_steps[best_idx]
                plt.axvline(x=best_step, color='gray', linestyle='--', alpha=0.6, label='Best Val')
            except Exception:
                pass
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (DDPM)')
        plt.grid(True, alpha=0.3)
        
        # Add a smoothed version if we have enough data points
        if len(self.losses) > 50:
            import numpy as np
            # Simple moving average
            window_size = min(50, len(self.losses) // 10)
            smoothed = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            steps = self.steps[window_size-1:]
            plt.plot(steps, smoothed, 'r-', linewidth=2, label=f'Train (smoothed={window_size})')
            if len(self.val_losses) > 10:
                v_window = min(20, max(5, len(self.val_losses) // 5))
                v_smoothed = np.convolve(self.val_losses, np.ones(v_window)/v_window, mode='valid')
                v_steps = self.val_steps[v_window-1:]
                plt.plot(v_steps, v_smoothed, 'g-', linewidth=2, label=f'Val (smoothed={v_window})')
            plt.legend()
        else:
            if len(self.val_losses) > 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the raw loss data
        loss_data = {
            'steps': self.steps,
            'losses': self.losses,
            'val_steps': self.val_steps,
            'val_losses': self.val_losses,
        }
        torch.save(loss_data, str(self.results_folder / 'training_loss_data.pt'))

    def plot_validation_only(self):
        if not self.accelerator.is_main_process or len(self.val_losses) == 0:
            return
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_steps, self.val_losses, color='orange', linewidth=1.8, label='Validation')
        try:
            best_idx = int(np.argmin(self.val_losses))
            best_step = self.val_steps[best_idx]
            best_val = self.val_losses[best_idx]
            plt.axvline(x=best_step, color='gray', linestyle='--', alpha=0.6, label='Best Val')
            plt.scatter([best_step], [best_val], color='red', zorder=3)
        except Exception:
            pass
        plt.xlabel('Training Step')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss (DDPM)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.results_folder / 'validation_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if data.get('scheduler') is not None and hasattr(self, 'scheduler'):
            try:
                self.scheduler.load_state_dict(data['scheduler'])
            except Exception as e:
                print(f"Warning: failed to load scheduler state: {e}")
        if data.get('ema') is not None:
            self.ema.load_state_dict(data["ema"])
        if 'best_val_loss' in data and data['best_val_loss'] is not None:
            self.best_val_loss = float(data['best_val_loss'])
        if 'best_val_step' in data and data['best_val_step'] is not None:
            self.best_val_step = int(data['best_val_step'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def _sync_ema_model(self):
        """
        Broadcast EMA weights from rank 0 to all ranks (helps keep validation consistent).
        """
        try:
            if not (dist.is_available() and dist.is_initialized()):
                return
            for v in self.ema.ema_model.state_dict().values():
                if isinstance(v, torch.Tensor):
                    dist.broadcast(v.data, src=0)
        except Exception:
            return

    def _init_nn_lookup(self, converter: BlockBiomeConverter | None = None):
        """
        Build a small embedding matrix for nearest-neighbor decoding using only
        block IDs present in the dataset mappings.
        """
        if getattr(self, "_nn_embedding_matrix", None) is not None and getattr(self, "_nn_block_ids", None) is not None:
            return

        emb_table = getattr(self.ds, "_emb_table", None)
        if emb_table is None or not isinstance(emb_table, torch.Tensor) or emb_table.dim() != 2:
            raise ValueError("[VoxelTrainerClassConditional] dataset is missing a valid _emb_table for NN decoding")

        candidate_block_ids = None
        present_block_ids = getattr(self.ds, "present_block_ids", None)
        if isinstance(present_block_ids, (list, tuple)) and len(present_block_ids) > 0:
            candidate_block_ids = list(present_block_ids)
        idx_to_block = getattr(self.ds, "index_to_block", None)
        if candidate_block_ids is None and isinstance(idx_to_block, torch.Tensor):
            candidate_block_ids = torch.unique(idx_to_block).tolist()
        elif candidate_block_ids is None and isinstance(idx_to_block, dict):
            candidate_block_ids = list({int(v) for v in idx_to_block.values()})
        elif candidate_block_ids is None and converter is not None:
            conv_idx = getattr(converter, "index_to_block", None)
            if isinstance(conv_idx, dict):
                candidate_block_ids = list({int(v) for v in conv_idx.values()})
            elif conv_idx is not None:
                candidate_block_ids = list({int(v) for v in list(conv_idx)})

        if not candidate_block_ids:
            raise ValueError("[VoxelTrainerClassConditional] could not determine candidate block IDs for NN decoding")

        candidate_block_ids = sorted({int(v) for v in candidate_block_ids})
        max_id = int(emb_table.shape[0]) - 1
        missing = [bid for bid in candidate_block_ids if not (0 <= int(bid) <= max_id)]
        if missing:
            show = missing[:20]
            raise ValueError(f"[VoxelTrainerClassConditional] missing embeddings for block IDs (out of range): {show}")

        block_ids = torch.tensor(candidate_block_ids, dtype=torch.long)
        emb_matrix = emb_table[block_ids].float().cpu()

        self._nn_block_ids = block_ids
        self._nn_embedding_matrix = emb_matrix
        self._nn_distance_metric = "l2"
        self._nn_chunk_size = 32768

        if self.accelerator.is_main_process:
            print(
                "[VoxelTrainerClassConditional] NN decode table ready: "
                f"unique_blocks={len(block_ids)}, "
                f"nn_table_shape={tuple(emb_matrix.shape)}, "
                f"full_emb_table_shape={tuple(emb_table.shape)}"
            )

    def _convert_emb_to_blocks(self, data, distance_metric: str | None = None, chunk_size: int | None = None):
        """
        Nearest-neighbor decode embeddings to block IDs using the trainer's
        precomputed, dataset-limited embedding matrix.
        """
        if getattr(self, "_nn_embedding_matrix", None) is None or getattr(self, "_nn_block_ids", None) is None:
            self._init_nn_lookup()

        metric = distance_metric or getattr(self, "_nn_distance_metric", "l2")
        chunk = int(chunk_size or getattr(self, "_nn_chunk_size", 32768))

        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        data = data.detach().cpu().float()

        has_batch = data.dim() == 5
        if has_batch:
            B, E, H, W, D = data.shape
            data_flat = data.permute(0, 2, 3, 4, 1).reshape(-1, E)
        else:
            E, H, W, D = data.shape
            data_flat = data.permute(1, 2, 3, 0).reshape(-1, E)

        emb_matrix = self._nn_embedding_matrix
        block_ids = self._nn_block_ids

        nearest_indices = torch.empty((data_flat.shape[0],), dtype=torch.long)
        for start in range(0, data_flat.shape[0], chunk):
            end = min(start + chunk, data_flat.shape[0])
            chunk_data = data_flat[start:end]
            if metric == "l2":
                distances = torch.cdist(chunk_data, emb_matrix)
                nearest = torch.argmin(distances, dim=1)
            elif metric == "cosine":
                data_norm = F.normalize(chunk_data, p=2, dim=1)
                emb_norm = F.normalize(emb_matrix, p=2, dim=1)
                sims = torch.mm(data_norm, emb_norm.t())
                nearest = torch.argmax(sims, dim=1)
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
            nearest_indices[start:end] = nearest

        decoded = block_ids[nearest_indices]
        if has_batch:
            return decoded.view(B, H, W, D)
        return decoded.view(H, W, D)

    @torch.inference_mode()
    def _distributed_sample_volumes(
        self,
        total_samples: int,
        class_labels: torch.Tensor,
        *,
        cond_scale: float,
        use_ema_model: bool = True,
    ):
        """
        DDPM analogue of MD4's _distributed_sample_indices:
        sample continuous volumes [B,C,D,H,W] across all ranks and gather to main.

        Returns:
          (samples_cpu, labels_cpu) on main process, (None, None) on others.
        """
        accelerator = self.accelerator
        device = accelerator.device

        total = int(total_samples)
        if not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        assert class_labels.dim() == 1 and class_labels.shape[0] == total, "class_labels must be length total_samples"

        # pick evaluation model
        if bool(use_ema_model) and hasattr(self, 'ema') and hasattr(self.ema, 'ema_model') and (self.ema.ema_model is not None):
            # sync EMA weights from rank0 so every rank samples the same model
            self._sync_ema_model()
            eval_model = self.ema.ema_model
        else:
            eval_model = accelerator.unwrap_model(self.model, keep_torch_compile=True)
        eval_model.eval()

        all_indices = list(range(total))
        local_samples = None
        local_labels = None

        with accelerator.split_between_processes(all_indices) as local_indices:
            local_n = len(local_indices)
            if local_n > 0:
                picked = [class_labels[i].item() for i in local_indices]
                local_labels = torch.tensor(picked, device=device, dtype=torch.long)
                # DDPM CFG sampler expects classes tensor
                local_samples = eval_model.sample(local_labels, cond_scale=float(cond_scale))
            else:
                # Empty rank: still participate in collectives with empty tensors
                C = int(getattr(self.model, 'channels', self.channels))
                D, H, W = self.image_size
                local_samples = torch.empty((0, C, D, H, W), dtype=torch.float32, device=device)
                local_labels = torch.empty((0,), dtype=torch.long, device=device)

        local_count = torch.tensor([local_samples.shape[0]], device=device)
        all_counts = accelerator.gather(local_count)

        # pad across processes and gather
        local_samples = accelerator.pad_across_processes(local_samples, dim=0, pad_index=0, pad_first=False)
        local_labels = accelerator.pad_across_processes(local_labels, dim=0, pad_index=0, pad_first=False)
        gathered_samples = accelerator.gather(local_samples)
        gathered_labels = accelerator.gather(local_labels)

        if accelerator.is_main_process:
            D, H, W = self.image_size
            C = gathered_samples.shape[1]
            gathered_samples = gathered_samples.view(accelerator.num_processes, -1, C, D, H, W)
            samples_list = [
                gathered_samples[i, :int(all_counts[i])]
                for i in range(accelerator.num_processes) if int(all_counts[i]) > 0
            ]
            gathered_samples = torch.cat(samples_list, dim=0).cpu() if samples_list else torch.empty((0, C, D, H, W), dtype=torch.float32)

            gathered_labels = gathered_labels.view(accelerator.num_processes, -1)
            labels_list = [
                gathered_labels[i, :int(all_counts[i])]
                for i in range(accelerator.num_processes) if int(all_counts[i]) > 0
            ]
            gathered_labels = torch.cat(labels_list, dim=0).cpu() if labels_list else torch.empty((0,), dtype=torch.long)
            accelerator.wait_for_everyone()
            return gathered_samples, gathered_labels
        else:
            accelerator.wait_for_everyone()
            return None, None

    @torch.inference_mode()
    def _compute_validation_loss(self, model_to_eval):
        if self.val_dl is None:
            return None
        accelerator = self.accelerator
        device = accelerator.device
        model_to_eval.eval()
        loss_sum = torch.tensor(0.0, device=device)
        n_items = torch.tensor(0, device=device, dtype=torch.long)
        total_batches = None
        try:
            total_batches = len(self.val_dl)
        except Exception:
            total_batches = None
        _tqdm_disable = not (accelerator.is_main_process and self.val_progress)
        _pbar = tqdm(total=total_batches, disable=_tqdm_disable, desc='Validating')
        for bi, batch in enumerate(self.val_dl):
            data, classes = batch
            data = data.to(device)
            classes = classes.to(device)
            
            # Embed on GPU
            data = self._embed_batch(data)
            
            with accelerator.autocast():
                loss = model_to_eval(data, classes=classes)
            bsz = data.shape[0]
            loss_sum += loss.detach() * bsz
            n_items += bsz
            if self.val_max_batches is not None and bi + 1 >= self.val_max_batches:
                if _pbar is not None:
                    _pbar.update(1)
                break
            if _pbar is not None:
                _pbar.update(1)
        if _pbar is not None:
            _pbar.close()
        loss_sum = accelerator.reduce(loss_sum, reduction='sum')
        n_items = accelerator.reduce(n_items, reduction='sum')
        model_to_eval.train()
        return (loss_sum / max(1, n_items)).item()

    def _embed_batch(self, block_ids):
        # block_ids: [B, X, Y, Z] LongTensor on device
        if self.device_emb_table is None:
             # Try to load it now (lazy load)
             if hasattr(self.ds, '_emb_table') and self.ds._emb_table is not None:
                  self.device_emb_table = self.ds._emb_table.to(self.accelerator.device)
             else:
                  raise ValueError("Dataset does not have _emb_table initialized, cannot embed on GPU.")

        # Embedding lookup
        # [B, X, Y, Z] -> [B, X, Y, Z, E]
        emb = F.embedding(block_ids, self.device_emb_table)
        
        # Permute to [B, E, Z, X, Y]
        # Input dims: 0:B, 1:X, 2:Y, 3:Z, 4:E
        # Target: B, E, Z, X, Y
        # 0->0, 4->1, 3->2, 1->3, 2->4
        return emb.permute(0, 4, 3, 1, 2).contiguous()

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        mappings = torch.load(self.mappings_file_path, weights_only=False)
        converter = BlockBiomeConverter(mappings['block_mappings'], mappings['biome_mappings'])
        if self.accelerator.is_main_process:
            print(f"[VoxelTrainerClassConditional] converter.block_to_emb is None? {converter.block_to_emb is None}")
        self._init_nn_lookup(converter)

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, classes = next(self.dl)  # Now returns tuple (block_ids, labels)
                    data = data.to(device)
                    classes = classes.to(device)
                    
                    # Embed on GPU
                    data = self._embed_batch(data)

                    # Diagnostic: log embedding stats on first step
                    if self.step == 0 and accelerator.is_main_process:
                        data_min, data_max = data.min().item(), data.max().item()
                        data_mean, data_std = data.mean().item(), data.std().item()
                        print(
                            f"[DDPM Data Range Check] Embedded data: "
                            f"min={data_min:.3f}, max={data_max:.3f}, "
                            f"mean={data_mean:.3f}, std={data_std:.3f}"
                        )
                        if data_min < -5 or data_max > 5:
                            print(
                                f"[DDPM Warning] Embedding range [{data_min:.2f}, {data_max:.2f}] is quite wide. "
                                f"If sampling produces NaN/Inf, consider enabling clip_denoised with appropriate clip_range."
                            )

                    with self.accelerator.autocast():
                        loss = self.model(data, classes=classes)  # Pass classes
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # Record loss for plotting
                if accelerator.is_main_process:
                    self.losses.append(total_loss)
                    self.steps.append(self.step)
                    if self.use_wandb:
                        try:
                            wandb.log({
                                "train/loss": total_loss,
                                "train/step": self.step,
                                "train/lr": self.scheduler.get_last_lr()[0],
                            }, step=self.step)
                        except Exception:
                            pass

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                accelerator.wait_for_everyone()

                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    milestone = self.step // self.save_and_sample_every
                    # global class labels: round-robin across classes
                    labels = torch.tensor([(i % int(self.num_classes)) for i in range(int(self.num_samples))], dtype=torch.long)
                    effective_scale = float(self.default_cond_scale) if self.default_cond_scale is not None else 4.0

                    # guided samples
                    all_volumes, all_labels = self._distributed_sample_volumes(
                        int(self.num_samples),
                        labels,
                        cond_scale=effective_scale,
                        use_ema_model=bool(self.evaluate_ema_model),
                    )

                    if accelerator.is_main_process and all_volumes is not None and all_labels is not None:
                        all_volumes_cat = all_volumes
                        all_labels_cat = all_labels

                        # Save the full 3D volume for later inspection
                        torch.save(all_volumes_cat, str(self.results_folder / f'sample-{milestone}.pt'))

                        print(f'Class labels: {all_labels_cat}')
                        classes_strs = converter.convert_class_to_biomes(all_labels_cat.to('cpu'))

                        # Converter expects embeddings laid out as [B, E, H, W, D].
                        # Sampling tensors are [B, E, D, H, W] -> permute for visualization.
                        vols_cpu = all_volumes_cat.to('cpu')
                        if vols_cpu.dim() == 5:
                            vols_cpu = vols_cpu.permute(0, 1, 3, 4, 2).contiguous()
                        converted_samples = self._convert_emb_to_blocks(vols_cpu)
                        save_chunks(
                            converted_samples,
                            self.step,
                            self.results_folder,
                            converter,
                            classes=classes_strs,
                            textured=True,
                            # textures_dir='block_textures/',
                        )

                        if self.use_wandb and self.wandb_log_images and self.accelerator.is_main_process:
                            try:
                                img_path = str(self.results_folder / f'sampled_chunks_ep_{self.step}.png')
                                wandb.log({f"samples/image_step_{self.step}": wandb.Image(img_path)}, step=self.step)
                            except Exception:
                                pass

                        if not self.save_only_last_checkpoint:
                            self.save(milestone)

                # periodic validation (MD4-style)
                if self.val_dl is not None and divisible_by(self.step, self.val_interval_steps):
                    with torch.inference_mode():
                        if self.evaluate_ema_model and hasattr(self, 'ema') and self.ema.ema_model is not None:
                            # Sync EMA weights before evaluating on all ranks
                            self._sync_ema_model()
                            model_to_eval = self.ema.ema_model
                        else:
                            model_to_eval = self.model
                        model_to_eval.eval()
                        val_loss = self._compute_validation_loss(model_to_eval)
                    if self.accelerator.is_main_process and val_loss is not None:
                        self.val_losses.append(float(val_loss))
                        self.val_steps.append(self.step)
                        if self.use_wandb:
                            try:
                                wandb.log({
                                    "val/loss": float(val_loss),
                                    "val/step": self.step,
                                }, step=self.step)
                            except Exception:
                                pass
                        logger.info(f"Validation loss at step {self.step}: {val_loss:.4f}")
                        self.plot_losses()
                        self.plot_validation_only()
                        if float(val_loss) < float(self.best_val_loss):
                            self.best_val_loss = float(val_loss)
                            self.best_val_step = int(self.step)
                            self._save_best_checkpoint()
                        if self.use_wandb:
                            try:
                                wandb.log({
                                    "val/best_loss": float(self.best_val_loss),
                                    "val/best_step": int(self.best_val_step) if self.best_val_step is not None else -1,
                                }, step=self.step)
                            except Exception:
                                pass
                        logger.info(f"Best validation so far: {self.best_val_loss:.4f} at step {self.best_val_step}")

                pbar.update(1)

        accelerator.print('training complete')
        if self.accelerator.is_main_process and self.save_only_last_checkpoint:
            self.save("final")
        
        # Generate final loss plot
        self.plot_losses()