import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Type


def one_param(m):
    return next(iter(m.parameters()))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return (torch.zeros(shape, device=device)
            .float().uniform_(0, 1) < prob)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    d3 = embed_dim // 3
    emb_x = get_1d_sincos_pos_embed_from_grid(d3, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(d3, grid[1])
    emb_z = get_1d_sincos_pos_embed_from_grid(d3, grid[2])
    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    gx = np.arange(grid_size, dtype=np.float32)
    gy = np.arange(grid_size, dtype=np.float32)
    gz = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(gx, gy, gz, indexing='ij')
    grid = np.stack(grid, axis=0)  # (3, g, g, g)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed  # (g*g*g, embed_dim)

# patchify in 3d


class PatchEmbed3D(nn.Module):
    def __init__(self, image_size=32, patch_size=1, in_dim=3, dim=768,
                 bias=True):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 3

        self.proj = nn.Conv3d(in_dim, dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)

    def forward(self, x):
        # x: [b, c, d, h, w]
        x = self.proj(x)  # [b, dim, g, g, g]
        x = x.flatten(2).transpose(1, 2)  # [b, t=g^3, dim]
        return x


class Attention(nn.Module):
    """Based on timm"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                'norm_layer must be provided if qk_norm or scale_norm is True')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# transformer block with adaLN


class DiTBlock3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop,)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c):
        s_msa, k_msa, g_msa, s_mlp, k_mlp, g_mlp = self.ada(c).chunk(6, dim=1)

        mod_x = modulate(self.norm1(x), s_msa, k_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(mod_x)
        x = self.proj_drop(x)

        x = x + g_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), s_mlp, k_mlp)
        )
        return x


class FinalLayer3D(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        self.head = nn.Linear(dim,
                              patch_size * patch_size * patch_size * out_dim,
                              bias=True)

    def forward(self, x, c):
        s, k = self.ada(c).chunk(2, dim=1)
        x = modulate(self.norm(x), s, k)
        x = self.head(x)
        return x

# base unconditional 3d diffusion transformer


class DiT3D(nn.Module):
    def __init__(self,
                 in_dim=64,
                 out_dim=64,
                 hidden_channels=768,
                 image_size=32,
                 patch_size=1,
                 time_dim=256,
                 is_ddim_sampling=False,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,
                 attn_drop=0.,
                 proj_drop=0.,
                 comb_method='add',
                 cond_in_dim=None,
                 class_conditional=False,
                 num_classes=None,
                 cond_drop_prob=0.5,
                 ):
        super().__init__()
        assert image_size % patch_size == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.time_dim = time_dim
        self.fourier_dim = time_dim // 4
        self.comb_method = comb_method
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        assert comb_method in ('add', 'cat')

        print(f'Initializing DiT3D class_conditional={class_conditional}')
        if class_conditional:
            assert num_classes is not None and num_classes > 1, (
                'num_classes must be provided for class-conditional models'
            )

        # for compatibility with lucid diffusion trainer
        self.is_ddim_sampling = is_ddim_sampling
        self.self_condition = False
        self.random_or_learned_sinusoidal_cond = None
        self.channels = in_dim

        self.patch_embed = PatchEmbed3D(image_size, patch_size,
                                        in_dim, hidden_channels, bias=True)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_channels),
            requires_grad=False
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(self.fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        cin = default(cond_in_dim, time_dim)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cin, hidden_channels)
        )

        self.blocks = nn.ModuleList([
            DiTBlock3D(hidden_channels, num_heads,
                       mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.final = FinalLayer3D(hidden_channels, patch_size, out_dim)

        if class_conditional:
            self.cond_drop_prob = cond_drop_prob
            self.num_classes = num_classes
            assert self.num_classes is not None and self.num_classes > 1, (
                'num_classes must be provided for class-conditional models'
            )
            self.null_class_emb = nn.Parameter(
                torch.randn(self.fourier_dim)
            )
            self.class_emb = nn.Embedding(num_classes, self.fourier_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(self.fourier_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim)
            )
        self._init_weights()

    def pos_encoding(self, t, channels):
        inv = 1.0 / (10000 ** (
            torch.arange(0, channels, 2,
                         device=one_param(self).device).float() / channels))
        a = torch.sin(t.repeat(1, channels // 2) * inv)
        b = torch.cos(t.repeat(1, channels // 2) * inv)
        return torch.cat([a, b], dim=-1)

    def time_encoding(self, t):
        t = self.pos_encoding(t, self.fourier_dim)
        t = self.time_mlp(t)
        return t

    def unpatchify(self, x_tok):
        # x_tok: [b, t, p^3*out_dim]
        b = x_tok.shape[0]
        p = self.patch_size
        g = self.image_size // self.patch_size
        c = self.out_dim

        x = x_tok.reshape(b, g, g, g, p, p, p, c)
        x = torch.einsum('bxyzpqrc->bcxpyqzr', x)
        x = x.reshape(b, c, g * p, g * p, g * p)
        return x

    def tokens_forward(self, x, cond_vec):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x, cond_vec)
        x = self.final(x, cond_vec)
        return x

    def forward(self, x, t, c=None, self_cond=None, cond_drop_prob=None):
        # x: [b, in_dim, d, h, w], t: [b]
        if x.dim() != 5:
            unsqueeze_input = True
            assert x.dim() == 4
            x = x.unsqueeze(0)
        else:
            unsqueeze_input = False
        t = t.unsqueeze(-1)
        t_vec = self.time_encoding(t)
        if self.class_conditional:
            drop = default(cond_drop_prob, self.cond_drop_prob)
            b = x.shape[0]
            keep = prob_mask_like((b,), 1 - drop, device=x.device)
            null = repeat(self.null_class_emb, 'd -> b d', b=b)
            c_raw = self.class_emb(c)

            c_raw = torch.where(rearrange(keep, 'b -> b 1'), c_raw, null)
            c_vec = self.class_mlp(c_raw)
            if self.comb_method == 'add':
                cond_in = t_vec + c_vec
            else:
                cond_in = torch.cat([t_vec, c_vec], dim=-1)
            cond_vec = self.cond_proj(cond_in)
        else:
            cond_vec = self.cond_proj(t_vec)
        toks = self.tokens_forward(x, cond_vec)
        out = self.unpatchify(toks)
        if unsqueeze_input:
            out = out.squeeze(0)
        return out

    def forward_with_cond_scale(self, x, t, c, cond_scale=1., **kwargs):
        logits = self.forward(x, t, c, cond_drop_prob=0.)
        if cond_scale == 1:
            return logits, None

        null_logits = self.forward(x, t, c, cond_drop_prob=1.)
        scaled = null_logits + cond_scale * (logits - null_logits)
        return scaled, null_logits

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        g = self.image_size // self.patch_size
        pe = get_3d_sincos_pos_embed(self.hidden_channels, g)
        self.pos_embed.data.copy_(
            torch.from_numpy(pe).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.class_conditional:
            w = self.class_emb.weight.data
            nn.init.xavier_uniform_(w)
        for blk in self.blocks:
            nn.init.constant_(blk.ada[-1].weight, 0)
            nn.init.constant_(blk.ada[-1].bias, 0)

        nn.init.constant_(self.final.ada[-1].weight, 0)
        nn.init.constant_(self.final.ada[-1].bias, 0)
        nn.init.constant_(self.final.head.weight, 0)
        nn.init.constant_(self.final.head.bias, 0)
