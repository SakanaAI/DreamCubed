"""
3D-native feature extractor for a voxel FID.

Render-FID pushes a 3D chunk through a 2D renderer and an ImageNet Inception
network: it is expensive, it depends on the texture pack and camera, and it only
sees the surfaces a camera happens to look at.  This module replaces that with a
small 3D CNN trained directly on block volumes; the Frechet distance is then
computed between the pooled features of real and generated chunks, with no
rendering anywhere in the loop.

Training objective
------------------
The Inception analogue would be plain classification, and biome classification
is the natural label space here.  On its own it is too weak: 15 biomes are
almost perfectly separable from the block histogram, so a classifier can reach
high accuracy while discarding every bit of spatial structure -- which would
make the resulting metric a slow re-derivation of the block-frequency statistics
we already compute elsewhere.

So the network is trained with two heads:

  1. biome classification (the Inception analogue), and
  2. corruption classification over structure-destroying, histogram-preserving
     corruptions (column shuffling, patch shuffling, vertical rolls, vertical
     flips, chunk splicing).

Head 2 is the important one: because the corruptions leave the block histogram
untouched, the only way to solve it is to encode geometry.  Corruptions are a
training-time objective only; features are always extracted from clean chunks.

Metric hygiene
--------------
  * Train on real chunks only -- never on generated data, or the metric is
    circular.
  * Train on the training split; compute reference statistics on a held-out
    split, and report the real-vs-real floor at the same sample count.
  * Use one frozen extractor for every comparison in the paper, including across
    data compositions.  The checkpoint stores its own config and a weight hash
    so a run can be tied to an exact feature space.
  * The feature dimension is deliberately 512, not Inception's 2048: the
    Frechet distance needs a d x d covariance estimated from N samples, and the
    small-sample bias grows with d.
"""

import hashlib
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import patch_voxels_to_volume, volume_to_patch_voxels

__version__ = "fid3d-0.1"

# index 0 is always 'clean'
CORRUPTION_TYPES = (
    'clean',
    'block_noise',
    'column_shuffle',
    'patch_shuffle',
    'vertical_roll',
    'vertical_flip',
    'half_swap',
    'horizontal_splice',
)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


def _groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class VoxelFeatureExtractor3D(nn.Module):
    """3D CNN over block indices, producing a pooled feature vector for FID.

    Volumes are [B, X, Y, Z] with Y the vertical axis.  The pooling stage
    averages over X and Z but keeps `y_bins` vertical bands, because vertical
    composition (bedrock below, terrain, then air) is exactly the kind of
    structure a voxel metric must not be blind to.  Pooling is adaptive, so the
    extractor also accepts volumes larger than the 32^3 it was trained on.
    """

    def __init__(self,
                 vocab_size,
                 num_biomes,
                 embed_dim=32,
                 channels=(64, 128, 256),
                 blocks_per_stage=(2, 2, 2),
                 stem_stride=2,
                 feature_channels=128,
                 y_bins=4,
                 width_mult=1.0,
                 num_corruptions=len(CORRUPTION_TYPES),
                 image_size=32,
                 ):
        super().__init__()
        assert len(channels) == len(blocks_per_stage)
        channels = tuple(max(8, int(round(c * width_mult))) for c in channels)
        feature_channels = max(8, int(round(feature_channels * width_mult)))

        self.vocab_size = int(vocab_size)
        self.num_biomes = int(num_biomes)
        self.num_corruptions = int(num_corruptions)
        self.embed_dim = int(embed_dim)
        self.channels = channels
        self.blocks_per_stage = tuple(blocks_per_stage)
        self.stem_stride = int(stem_stride)
        self.feature_channels = feature_channels
        self.y_bins = int(y_bins)
        self.image_size = int(image_size)
        self.feature_dim = feature_channels * self.y_bins

        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.stem = ConvGNAct(embed_dim, channels[0], stride=stem_stride)

        stages = []
        for i, ch in enumerate(channels):
            layers = [ResBlock3D(ch) for _ in range(blocks_per_stage[i])]
            if i < len(channels) - 1:
                layers.append(ConvGNAct(ch, channels[i + 1], stride=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

        self.proj = ConvGNAct(channels[-1], feature_channels, kernel_size=1)
        self.biome_head = nn.Linear(self.feature_dim, self.num_biomes)
        # num_corruptions=0 builds the classifier-only ablation: same trunk, same
        # feature dimension, no structural objective
        self.corruption_head = (nn.Linear(self.feature_dim, self.num_corruptions)
                                if self.num_corruptions > 0 else None)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def features(self, x_indices: torch.Tensor) -> torch.Tensor:
        """[B, X, Y, Z] block indices -> [B, feature_dim]."""
        x = x_indices.long().clamp(0, self.vocab_size - 1)
        h = self.embed(x).permute(0, 4, 1, 2, 3).contiguous()
        h = self.stem(h)
        for stage in self.stages:
            h = stage(h)
        h = self.proj(h)
        h = F.adaptive_avg_pool3d(h, (1, self.y_bins, 1))
        return h.flatten(1)

    def forward(self, x_indices):
        feats = self.features(x_indices)
        corr = None if self.corruption_head is None else self.corruption_head(feats)
        return feats, self.biome_head(feats), corr

    def config(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'num_biomes': self.num_biomes,
            'embed_dim': self.embed_dim,
            'channels': list(self.channels),
            'blocks_per_stage': list(self.blocks_per_stage),
            'stem_stride': self.stem_stride,
            'feature_channels': self.feature_channels,
            'y_bins': self.y_bins,
            'num_corruptions': self.num_corruptions,
            'image_size': self.image_size,
            'feature_dim': self.feature_dim,
        }

    def weight_hash(self) -> str:
        h = hashlib.sha256()
        for name, param in sorted(self.state_dict().items()):
            h.update(name.encode())
            h.update(param.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()[:16]


def save_feature_extractor(model, path, extra: Optional[Dict] = None):
    payload = {
        'version': __version__,
        'config': model.config(),
        'model': model.state_dict(),
        'weight_hash': model.weight_hash(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    return payload['weight_hash']


def load_feature_extractor(path, device='cuda', strict=True):
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    cfg = dict(payload['config'])
    cfg.pop('feature_dim', None)
    model = VoxelFeatureExtractor3D(**cfg)
    model.load_state_dict(payload['model'], strict=strict)
    model.eval().requires_grad_(False).to(device)
    return model, payload


# ---------------------------------------------------------------------------
# structure-destroying, histogram-preserving corruptions
# ---------------------------------------------------------------------------


def _per_sample_permutation(b: int, n: int, device) -> torch.Tensor:
    return torch.argsort(torch.rand((b, n), device=device), dim=1)


def block_noise(x: torch.Tensor, severity: float = 0.05) -> torch.Tensor:
    """Replace a fraction of voxels with blocks resampled from the same batch."""
    b = x.shape[0]
    flat = x.reshape(-1)
    idx = torch.randint(0, flat.numel(), x.shape, device=x.device)
    replacement = flat[idx.reshape(-1)].reshape(x.shape)
    mask = torch.rand(x.shape, device=x.device) < float(severity)
    return torch.where(mask, replacement, x)


def column_shuffle(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    """Permute whole vertical columns. Exactly preserves the block histogram."""
    b, sx, sy, sz = x.shape
    cols = x.permute(0, 2, 1, 3).reshape(b, sy, sx * sz)
    perm = _per_sample_permutation(b, sx * sz, x.device)
    cols = torch.gather(cols, 2, perm.unsqueeze(1).expand(-1, sy, -1))
    return cols.reshape(b, sy, sx, sz).permute(0, 2, 1, 3).contiguous()


def _partner_index(b: int, device) -> Optional[torch.Tensor]:
    """A random cyclic shift, so no sample is ever spliced with itself."""
    if b < 2:
        return None
    shift = int(torch.randint(1, b, (1,), device='cpu').item())
    return (torch.arange(b, device=device) + shift) % b


def patch_shuffle(x: torch.Tensor, severity: float = 8.0) -> torch.Tensor:
    """Permute k^3 sub-cubes. Preserves local texture, destroys global layout."""
    assert x.shape[1] == x.shape[2] == x.shape[3], 'patch_shuffle expects a cubic volume'
    k = int(severity)
    if x.shape[1] % k != 0:
        k = 8 if x.shape[1] % 8 == 0 else 4
    patches = volume_to_patch_voxels(x, k)
    perm = _per_sample_permutation(x.shape[0], patches.shape[1], x.device)
    patches = torch.gather(patches, 1, perm.unsqueeze(-1).expand(-1, -1, patches.shape[2]))
    return patch_voxels_to_volume(patches, k)


def vertical_roll(x: torch.Tensor, severity: float = 4.0) -> torch.Tensor:
    """Roll each column independently along Y: floating and buried terrain."""
    b, sx, sy, sz = x.shape
    max_shift = max(1, int(severity))
    shift = torch.randint(-max_shift, max_shift + 1, (b, sx, 1, sz), device=x.device)
    ys = torch.arange(sy, device=x.device).view(1, 1, sy, 1)
    idx = (ys - shift) % sy
    return torch.gather(x, 2, idx.expand(b, sx, sy, sz))


def vertical_flip(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    """Upside-down terrain. Exactly preserves the block histogram."""
    return torch.flip(x, dims=(2,))


def half_swap(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    """Take the upper half from a different chunk: implausible vertical seam."""
    partner = _partner_index(x.shape[0], x.device)
    if partner is None:
        return x
    cut = x.shape[2] // 2
    return torch.cat([x[:, :, :cut], x[partner][:, :, cut:]], dim=2)


def horizontal_splice(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    """Glue half of another chunk along X: the seam artifact outpainting makes."""
    partner = _partner_index(x.shape[0], x.device)
    if partner is None:
        return x
    cut = x.shape[1] // 2
    return torch.cat([x[:, :cut], x[partner][:, cut:]], dim=1)


CORRUPTION_FNS = {
    'block_noise': block_noise,
    'column_shuffle': column_shuffle,
    'patch_shuffle': patch_shuffle,
    'vertical_roll': vertical_roll,
    'vertical_flip': vertical_flip,
    'half_swap': half_swap,
    'horizontal_splice': horizontal_splice,
}

DEFAULT_SEVERITIES = {
    'block_noise': 0.05,
    'column_shuffle': 1.0,
    'patch_shuffle': 8.0,
    'vertical_roll': 4.0,
    'vertical_flip': 1.0,
    'half_swap': 1.0,
    'horizontal_splice': 1.0,
}


def apply_corruptions(x: torch.Tensor, corruption_prob: float,
                      types: Sequence[str], severities: Optional[Dict] = None):
    """Corrupt a random subset of the batch.

    Returns (corrupted volumes, corruption label per sample) where label 0 means
    the sample was left clean.
    """
    severities = severities or DEFAULT_SEVERITIES
    b = x.shape[0]
    labels = torch.zeros((b,), dtype=torch.long, device=x.device)
    if corruption_prob <= 0 or not types:
        return x, labels

    selected = torch.rand((b,), device=x.device) < float(corruption_prob)
    if not bool(selected.any()):
        return x, labels

    choice = torch.randint(0, len(types), (b,), device=x.device)
    out = x.clone()
    for ti, name in enumerate(types):
        rows = selected & (choice == ti)
        if not bool(rows.any()):
            continue
        # corrupt the whole batch and keep the selected rows: the splicing
        # corruptions need partners, and a one-row subset has none
        corrupted = CORRUPTION_FNS[name](x, severities.get(name, 1.0))
        out[rows] = corrupted[rows]
        labels[rows] = CORRUPTION_TYPES.index(name)
    return out, labels


# ---------------------------------------------------------------------------
# the metric itself
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_features(model, volumes: Iterable, batch_size: int = 64,
                     device='cuda', autocast: bool = True,
                     progress: bool = False) -> np.ndarray:
    """Featurize an iterable/tensor of [*, X, Y, Z] index volumes."""
    model.eval()
    if isinstance(volumes, torch.Tensor):
        chunks = torch.split(volumes, batch_size)
    else:
        chunks = volumes

    feats = []
    iterator = chunks
    if progress:
        from tqdm import tqdm
        iterator = tqdm(chunks, desc='features')
    for batch in iterator:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        is_cuda = str(device).startswith('cuda')
        with torch.autocast(device_type=('cuda' if is_cuda else 'cpu'),
                            dtype=torch.bfloat16, enabled=(autocast and is_cuda)):
            f = model.features(batch)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0).numpy()


@torch.inference_mode()
def histogram_features(volumes: torch.Tensor, vocab_size: int, batch_size: int = 256,
                       device='cpu') -> np.ndarray:
    """Per-chunk normalized block frequencies: the null hypothesis for this metric.

    Any structural corruption that preserves the block histogram is invisible
    here by construction, which makes this a useful control -- if the learned
    features rank models the same way these do, the network has not learned
    anything beyond block statistics.
    """
    out = []
    for batch in torch.split(volumes, batch_size):
        x = batch.to(device).reshape(batch.shape[0], -1).long()
        counts = torch.zeros((x.shape[0], vocab_size), dtype=torch.float32, device=x.device)
        counts.scatter_add_(1, x, torch.ones_like(x, dtype=torch.float32))
        out.append((counts / x.shape[1]).cpu())
    return torch.cat(out, dim=0).numpy()


def compute_feature_stats(features: np.ndarray):
    features = np.asarray(features, dtype=np.float64)
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Standard Frechet distance between two Gaussians (same formula as FID)."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_features(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    mu_a, sigma_a = compute_feature_stats(feats_a)
    mu_b, sigma_b = compute_feature_stats(feats_b)
    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


if __name__ == '__main__':
    torch.manual_seed(0)
    v, biomes = 171, 15
    model = VoxelFeatureExtractor3D(vocab_size=v, num_biomes=biomes)
    print('feature dim', model.feature_dim)
    print('params %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    x = torch.randint(0, v, (4, 32, 32, 32))
    feats, biome_logits, corr_logits = model(x)
    print('features', tuple(feats.shape), 'biome', tuple(biome_logits.shape),
          'corruption', tuple(corr_logits.shape))

    # every corruption must preserve the shape, and all but the splicing ones
    # must leave the block histogram untouched
    hist = torch.bincount(x.reshape(-1), minlength=v)
    for name, fn in CORRUPTION_FNS.items():
        y = fn(x, DEFAULT_SEVERITIES[name])
        assert y.shape == x.shape, name
        if name in ('column_shuffle', 'patch_shuffle', 'vertical_flip', 'vertical_roll'):
            assert torch.equal(torch.bincount(y.reshape(-1), minlength=v), hist), name
        assert not torch.equal(y, x), name
    print('corruptions ok (histogram preserved where expected)')

    # a larger volume must still featurize, thanks to adaptive pooling
    big = torch.randint(0, v, (1, 64, 32, 64))
    print('64x32x64 features', tuple(model.features(big).shape))

    a = np.random.randn(2048, model.feature_dim)
    b = np.random.randn(2048, model.feature_dim) + 0.1
    print('sanity FID(a,a)=%.4f  FID(a,b)=%.4f' % (fid_from_features(a, a),
                                                   fid_from_features(a, b)))
