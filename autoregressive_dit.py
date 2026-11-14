"""
Autoregressive baseline for Dream-Cubed.

Reuses the same biome-conditional 3D DiT backbone as the MD4 / DDPM models
(`dit.DiT3D`) so that the only difference between the three generative
formulations is the probabilistic model on top of it:

    MD4   : bidirectional attention, masked-diffusion denoiser, factorized over
            all voxels conditioned on the partially unmasked volume.
    DDPM  : bidirectional attention, v-prediction in embedding space.
    AR    : causal attention over patch tokens in a fixed raster order, with an
            exact chain-rule factorization over every voxel.

The backbone (patch embedding, 3D sin-cos positional embedding, adaLN-Zero
blocks, learned biome embedding + null embedding for classifier-free guidance)
is component-for-component identical to `DiT3D`; only the attention mask, the
input shift and the output head differ.  Because there is no diffusion time,
the timestep pathway is retained but held at t = 0 so the conditioning stack and
parameter count stay comparable to the diffusion models.

Two output heads are supported:

    local_head_depth == 0
        The `FinalLayer3D` head of `DiT3D` predicts all p^3 voxels of a patch at
        once.  Voxels *inside* a patch are then conditionally independent given
        the prefix, i.e. this variant has the same joint-distribution weakness
        that masked diffusion has, just at patch granularity.  Useful as an
        ablation, not as the headline baseline.

    local_head_depth > 0  (default)
        A small causal transformer decodes the p^3 voxels of a patch one at a
        time, conditioned on the global hidden state.  The resulting model is a
        proper autoregressive model over all 32^3 voxels with no conditional
        independence assumptions anywhere.

Voxel layout.  Dataset volumes are indexed [X, Y, Z] with axis 1 the vertical
(Minecraft Y) axis -- see `data_utils.rotate_voxels_90_fix_stairs_torch`, which
rotates axes (0, 2).  The default generation order is therefore bottom-up:
Y slowest, then Z, then X, both across patches and within a patch.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from data_utils import patch_voxels_to_volume, volume_to_patch_voxels
from dit import (
    Attention,
    DiTBlock3D,
    FinalLayer3D,
    PatchEmbed3D,
    default,
    get_3d_sincos_pos_embed,
    modulate,
    one_param,
    prob_mask_like,
)


# ---------------------------------------------------------------------------
# volume <-> (patch token, voxel-in-patch) layout helpers
# ---------------------------------------------------------------------------
#
# `PatchEmbed3D` is fed the volume permuted to [B, C, Z, X, Y] (see how MD4
# builds its input), so the conv token at flat index t corresponds to grid
# coordinates (zg, xg, yg) with t = (zg * g + xg) * g + yg.  This module and the
# `volume_to_patch_voxels` / `patch_voxels_to_volume` pair in `data_utils` are
# all expressed in that "conv token order"; the reordering into the
# autoregressive order happens once, via `ar_perm`.


def build_raster_permutation(grid: int, order: str = 'y_slow') -> torch.Tensor:
    """Permutation from autoregressive token index -> conv token index."""
    idx = torch.arange(grid ** 3, dtype=torch.long)
    zg = idx // (grid * grid)
    xg = (idx // grid) % grid
    yg = idx % grid
    if order == 'y_slow':          # bottom-up: fill a horizontal layer, then rise
        key = (yg * grid + zg) * grid + xg
    elif order == 'conv':          # native flatten order of the patch embedding
        key = idx
    elif order == 'x_slow':
        key = (xg * grid + zg) * grid + yg
    else:
        raise ValueError(f'unknown token order: {order}')
    return torch.argsort(key)


# ---------------------------------------------------------------------------
# causal building blocks (parameter-identical to their bidirectional versions)
# ---------------------------------------------------------------------------


class CausalAttention(Attention):
    """`dit.Attention` with a causal mask and an optional incremental KV cache.

    Parameter names and shapes are unchanged, so a state dict is interchangeable
    with the bidirectional backbone.
    """

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if cache is not None:
            if cache.get('k') is not None:
                k = torch.cat((cache['k'], k), dim=2)
                v = torch.cat((cache['v'], v), dim=2)
            cache['k'], cache['v'] = k, v

        # With a populated cache the query is a suffix of the key sequence, and
        # SDPA's `is_causal` aligns the mask top-left, so it only applies when
        # query and key lengths match.
        is_causal = (q.shape[2] == k.shape[2]) and q.shape[2] > 1

        x = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalDiTBlock3D(DiTBlock3D):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio,
                         attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn = CausalAttention(dim, num_heads=num_heads, qkv_bias=True,
                                    attn_drop=attn_drop)

    def forward(self, x, c, cache=None):
        s_msa, k_msa, g_msa, s_mlp, k_mlp, g_mlp = self.ada(c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), s_msa, k_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(mod_x, cache=cache)
        x = self.proj_drop(x)

        x = x + g_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), s_mlp, k_mlp)
        )
        return x


class LocalARHead(nn.Module):
    """Small causal transformer that decodes the p^3 voxels inside one patch.

    Runs over a flattened batch of (batch x patch) sequences during training, so
    the whole volume is still one parallel forward pass.
    """

    def __init__(self, vocab_size, patch_size, ctx_dim, dim=384, depth=2,
                 num_heads=6, mlp_ratio=4.0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(patch_size) ** 3
        self.dim = int(dim)
        self.bos_id = int(vocab_size)

        self.tok_emb = nn.Embedding(vocab_size + 1, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, dim))
        self.ctx_proj = nn.Linear(ctx_dim, dim)
        self.cond_proj = nn.Linear(ctx_dim, dim)
        self.blocks = nn.ModuleList([
            CausalDiTBlock3D(
                dim, num_heads, mlp_ratio, attn_drop, proj_drop,
            )
            for _ in range(depth)
        ])
        self.final = FinalLayer3D(dim, 1, vocab_size)

    def prepare(self, h, cond_vec):
        """Project the global state / conditioning into the local width."""
        return self.ctx_proj(h), self.cond_proj(cond_vec)


    def forward(self, ctx, cond, prev_tokens, cache=None, pos_offset=0):
        """ctx, cond: [M, dim]; prev_tokens: [M, L'] -> logits [M, L', V]."""
        z = self.tok_emb(prev_tokens)
        n = z.shape[1]
        z = z + self.pos_emb[:, pos_offset:pos_offset + n] + ctx.unsqueeze(1)
        for i, blk in enumerate(self.blocks):
            blk_cache = None if cache is None else cache[i]
            z = blk(z, cond, cache=blk_cache)
        return self.final(z, cond)


# ---------------------------------------------------------------------------
# autoregressive DiT
# ---------------------------------------------------------------------------


class ARDiT3D(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_channels=768,
                 image_size=32,
                 patch_size=4,
                 time_dim=256,
                 depth=25,
                 num_heads=8,
                 mlp_ratio=4.0,
                 attn_drop=0.,
                 proj_drop=0.,
                 comb_method='add',
                 cond_in_dim=None,
                 class_conditional=True,
                 num_classes=None,
                 cond_drop_prob=0.2,
                 local_head_depth=2,
                 local_head_dim=384,
                 local_head_heads=6,
                 token_order='y_slow',
                 use_gradient_checkpointing=False,
                 ):
        super().__init__()
        assert image_size % patch_size == 0
        self.vocab_size = int(vocab_size)
        self.hidden_channels = hidden_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_tokens = self.grid ** 3
        self.patch_voxels = patch_size ** 3
        self.time_dim = time_dim
        self.fourier_dim = time_dim // 4
        self.comb_method = comb_method
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        self.token_order = token_order
        self.use_gradient_checkpointing = use_gradient_checkpointing
        assert comb_method in ('add', 'cat')

        if class_conditional:
            assert num_classes is not None and num_classes > 1, (
                'num_classes must be provided for class-conditional models'
            )

        # The extra input channel mirrors the mask channel of the MD4 model so
        # the patch embedding has exactly the same shape; it is reserved for
        # "not generated yet" and unused on the default path.
        self.patch_embed = PatchEmbed3D(image_size, patch_size,
                                        self.vocab_size + 1, hidden_channels,
                                        bias=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, hidden_channels),
            requires_grad=False
        )
        self.bos_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))

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
            CausalDiTBlock3D(hidden_channels, num_heads,
                             mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])

        self.local_head_depth = int(local_head_depth)
        if self.local_head_depth > 0:
            self.final = None
            self.local_head = LocalARHead(
                vocab_size=self.vocab_size,
                patch_size=patch_size,
                ctx_dim=hidden_channels,
                dim=local_head_dim,
                depth=self.local_head_depth,
                num_heads=local_head_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            self.final = FinalLayer3D(hidden_channels, patch_size, self.vocab_size)
            self.local_head = None

        if class_conditional:
            self.null_class_emb = nn.Parameter(torch.randn(self.fourier_dim))
            self.class_emb = nn.Embedding(num_classes, self.fourier_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(self.fourier_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim)
            )

        perm = build_raster_permutation(self.grid, token_order)
        self.register_buffer('ar_perm', perm, persistent=False)
        self.register_buffer('ar_inv', torch.argsort(perm), persistent=False)

        self._init_weights()

    # -- conditioning -------------------------------------------------------

    def pos_encoding(self, t, channels):
        inv = 1.0 / (10000 ** (
            torch.arange(0, channels, 2,
                         device=one_param(self).device).float() / channels))
        a = torch.sin(t.repeat(1, channels // 2) * inv)
        b = torch.cos(t.repeat(1, channels // 2) * inv)
        return torch.cat([a, b], dim=-1)

    def cond_vector(self, batch_size, class_cond=None, cond_drop_prob=None,
                    device=None):
        device = device if device is not None else one_param(self).device
        # No diffusion time in the AR formulation; the timestep pathway is kept
        # at zero so the conditioning stack matches the diffusion baselines.
        t = torch.zeros((batch_size, 1), device=device)
        t_vec = self.time_mlp(self.pos_encoding(t, self.fourier_dim))

        if not self.class_conditional:
            return self.cond_proj(t_vec)

        assert class_cond is not None, 'class-conditional model requires labels'
        drop = default(cond_drop_prob, self.cond_drop_prob)
        keep = prob_mask_like((batch_size,), 1 - drop, device=device)
        null = repeat(self.null_class_emb, 'd -> b d', b=batch_size)
        c_raw = self.class_emb(class_cond)
        c_raw = torch.where(rearrange(keep, 'b -> b 1'), c_raw, null)
        c_vec = self.class_mlp(c_raw)
        if self.comb_method == 'add':
            cond_in = t_vec + c_vec
        else:
            cond_in = torch.cat([t_vec, c_vec], dim=-1)
        return self.cond_proj(cond_in)


    # -- layout -------------------------------------------------------------

    def targets_from_volume(self, x_indices: torch.Tensor) -> torch.Tensor:
        """[B, S, S, S] -> [B, T, p^3] in autoregressive order."""
        xv = volume_to_patch_voxels(x_indices, self.patch_size)
        return xv[:, self.ar_perm]

    def volume_from_targets(self, xv: torch.Tensor) -> torch.Tensor:
        """[B, T, p^3] in autoregressive order -> [B, S, S, S]."""
        return patch_voxels_to_volume(xv[:, self.ar_inv], self.patch_size)

    def _embed_volume(self, x_indices: torch.Tensor) -> torch.Tensor:
        """[B, S, S, S] index volume -> [B, T, dim] patch tokens (conv order)."""
        oh = F.one_hot(x_indices.long(), num_classes=self.vocab_size + 1)
        oh = oh.permute(0, 4, 1, 2, 3).float()      # [B, V+1, X, Y, Z]
        oh = oh.permute(0, 1, 4, 2, 3)              # [B, V+1, Z, X, Y]
        return self.patch_embed(oh)

    def _embed_patch(self, voxels: torch.Tensor) -> torch.Tensor:
        """[B, p^3] voxels of a single patch (AR order) -> [B, dim]."""
        p = self.patch_size
        v = voxels.reshape(-1, p, p, p)             # yp zp xp
        v = v.permute(0, 2, 3, 1)                   # zp xp yp (conv layout)
        oh = F.one_hot(v.long(), num_classes=self.vocab_size + 1)
        oh = oh.permute(0, 4, 1, 2, 3).float()
        return self.patch_embed.proj(oh).flatten(1)

    # -- backbone -----------------------------------------------------------

    def backbone(self, tokens, cond_vec, caches=None):
        x = tokens
        for i, blk in enumerate(self.blocks):
            if caches is None and self.use_gradient_checkpointing and self.training:
                x = checkpoint(blk, x, cond_vec, use_reentrant=False)
            else:
                x = blk(x, cond_vec, cache=None if caches is None else caches[i])
        return x

    def forward(self, x_indices, class_cond=None, cond_drop_prob=None):
        """Teacher-forced forward pass.

        Returns (logits [B, T, p^3, V], targets [B, T, p^3]).
        """
        x_indices = x_indices.long()
        b = x_indices.shape[0]
        device = x_indices.device

        cond_vec = self.cond_vector(b, class_cond, cond_drop_prob, device)

        tokens = self._embed_volume(x_indices)[:, self.ar_perm]
        bos = self.bos_token.to(tokens.dtype).expand(b, -1, -1)
        shifted = torch.cat([bos, tokens[:, :-1]], dim=1)
        # Slot i holds the content of patch i-1 and the position of patch i,
        # i.e. the position that is being predicted.
        shifted = shifted + self.pos_embed[:, self.ar_perm]

        h = self.backbone(shifted, cond_vec)
        targets = self.targets_from_volume(x_indices)

        if self.local_head is None:
            logits = self.final(h, cond_vec)
            logits = logits.reshape(b, self.num_tokens, self.patch_voxels, self.vocab_size)
            return logits, targets

        ctx, cond = self.local_head.prepare(h, cond_vec)
        m = b * self.num_tokens
        ctx = ctx.reshape(m, -1)
        cond = cond.unsqueeze(1).expand(-1, self.num_tokens, -1).reshape(m, -1)

        flat_targets = targets.reshape(m, self.patch_voxels)
        bos = torch.full((m, 1), self.local_head.bos_id, dtype=torch.long, device=device)
        prev = torch.cat([bos, flat_targets[:, :-1]], dim=1)

        if self.use_gradient_checkpointing and self.training:
            logits = checkpoint(self.local_head, ctx, cond, prev, use_reentrant=False)
        else:
            logits = self.local_head(ctx, cond, prev)
        logits = logits.reshape(b, self.num_tokens, self.patch_voxels, self.vocab_size)
        return logits, targets

    def forward_with_cond_scale(self, x_indices, class_cond, cond_scale=1., **kwargs):
        """Teacher-forced CFG logits (for scoring; sampling guides per voxel)."""
        logits, targets = self.forward(x_indices, class_cond, cond_drop_prob=0.)
        if cond_scale == 1:
            return logits, targets
        null_logits, _ = self.forward(x_indices, class_cond, cond_drop_prob=1.)
        return null_logits + cond_scale * (logits - null_logits), targets

    # -- likelihood ---------------------------------------------------------

    def nats_per_voxel(self, x_indices, class_cond=None, cond_drop_prob=0.0):
        """Exact negative log-likelihood in nats per voxel."""
        logits, targets = self.forward(x_indices, class_cond, cond_drop_prob)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size).float(),
            targets.reshape(-1),
            reduction='mean',
        )

    # -- sampling -----------------------------------------------------------

    @torch.inference_mode()
    def generate(self,
                 batch_size,
                 class_cond=None,
                 cond_scale=1.0,
                 temperature=1.0,
                 top_k=0,
                 top_p=0.0,
                 allowed_block_mask=None,
                 known_volume=None,
                 known_mask=None,
                 progress=False,
                 device=None):
        """Sample volumes of shape [B, S, S, S] with integer block indices.

        Args:
            class_cond: [B] biome labels (required for class-conditional models).
            cond_scale: classifier-free guidance scale, applied per voxel.
            allowed_block_mask: optional bool [V] or [B, V]; False entries are
                forbidden everywhere (hard vocabulary constraints).
            known_volume / known_mask: optional [B, S, S, S] volume and bool mask
                of voxels that are pinned to user-authored values.  Pinned voxels
                are fed back into the context, so any voxel that comes *later* in
                the raster order is conditioned on them.  Constraints that lie
                ahead of the current position cannot be conditioned on -- that
                asymmetry is inherent to autoregressive models and is exactly
                what masked diffusion avoids.
        """
        device = device if device is not None else one_param(self).device
        b = int(batch_size)
        t_total = self.num_tokens
        lseq = self.patch_voxels

        use_cfg = (self.class_conditional and class_cond is not None
                   and float(cond_scale) != 1.0)

        if self.class_conditional:
            assert class_cond is not None
            class_cond = class_cond.to(device).long()
            cond_vec = self.cond_vector(b, class_cond, cond_drop_prob=0., device=device)
            if use_cfg:
                null_vec = self.cond_vector(b, class_cond, cond_drop_prob=1., device=device)
                cond_vec = torch.cat([cond_vec, null_vec], dim=0)
        else:
            cond_vec = self.cond_vector(b, None, None, device)

        known_pv = known_mask_pv = None
        if known_volume is not None:
            assert known_mask is not None, 'known_volume requires known_mask'
            known_pv = self.targets_from_volume(known_volume.to(device).long())
            known_mask_pv = self.targets_from_volume(known_mask.to(device).long()).bool()

        caches = [{} for _ in self.blocks]
        out = torch.zeros((b, t_total, lseq), dtype=torch.long, device=device)
        prev_emb = self.bos_token.expand(b, -1, -1).squeeze(1)

        rng = tqdm(range(t_total), disable=not progress, desc='AR sampling')
        for i in rng:
            tok = prev_emb + self.pos_embed[0, self.ar_perm[i]]
            tok = torch.cat([tok, tok], dim=0) if use_cfg else tok

            x = tok.unsqueeze(1)
            for blk, cache in zip(self.blocks, caches):
                x = blk(x, cond_vec, cache=cache)
            h = x[:, 0]

            known_i = None if known_pv is None else known_pv[:, i]
            known_mask_i = None if known_mask_pv is None else known_mask_pv[:, i]

            voxels = self._decode_patch(
                h, cond_vec, b, use_cfg, float(cond_scale),
                temperature=temperature, top_k=top_k, top_p=top_p,
                allowed_block_mask=allowed_block_mask,
                known=known_i, known_mask=known_mask_i,
            )
            out[:, i] = voxels
            prev_emb = self._embed_patch(voxels)

        return self.volume_from_targets(out)

    def _decode_patch(self, h, cond_vec, b, use_cfg, cond_scale, *,
                      temperature, top_k, top_p, allowed_block_mask,
                      known, known_mask):
        lseq = self.patch_voxels

        if self.local_head is None:
            logits = self.final(h.unsqueeze(1), cond_vec)[:, 0]
            logits = logits.reshape(-1, lseq, self.vocab_size)
            if use_cfg:
                cond_l, null_l = logits[:b], logits[b:]
                logits = null_l + cond_scale * (cond_l - null_l)
            voxels = sample_from_logits(logits, temperature, top_k, top_p,
                                        allowed_block_mask)
            if known_mask is not None:
                voxels = torch.where(known_mask, known, voxels)
            return voxels

        ctx, cond = self.local_head.prepare(h, cond_vec)
        caches = [{} for _ in self.local_head.blocks]
        prev = torch.full((ctx.shape[0], 1), self.local_head.bos_id,
                          dtype=torch.long, device=h.device)
        voxels = torch.zeros((b, lseq), dtype=torch.long, device=h.device)

        for j in range(lseq):
            logits = self.local_head(ctx, cond, prev, cache=caches, pos_offset=j)[:, 0]
            if use_cfg:
                cond_l, null_l = logits[:b], logits[b:]
                logits = null_l + cond_scale * (cond_l - null_l)
            vox = sample_from_logits(logits, temperature, top_k, top_p,
                                     allowed_block_mask)
            if known_mask is not None:
                vox = torch.where(known_mask[:, j], known[:, j], vox)
            voxels[:, j] = vox
            prev = torch.cat([vox, vox], dim=0).unsqueeze(1) if use_cfg else vox.unsqueeze(1)

        return voxels

    # -- init ---------------------------------------------------------------

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pe = get_3d_sincos_pos_embed(self.hidden_channels, self.grid)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.bos_token, std=0.02)

        if self.class_conditional:
            nn.init.xavier_uniform_(self.class_emb.weight.data)

        for blk in self.blocks:
            nn.init.constant_(blk.ada[-1].weight, 0)
            nn.init.constant_(blk.ada[-1].bias, 0)

        if self.final is not None:
            nn.init.constant_(self.final.ada[-1].weight, 0)
            nn.init.constant_(self.final.ada[-1].bias, 0)
            nn.init.constant_(self.final.head.weight, 0)
            nn.init.constant_(self.final.head.bias, 0)

        if self.local_head is not None:
            nn.init.normal_(self.local_head.tok_emb.weight, std=0.02)
            nn.init.normal_(self.local_head.pos_emb, std=0.02)
            for blk in self.local_head.blocks:
                nn.init.constant_(blk.ada[-1].weight, 0)
                nn.init.constant_(blk.ada[-1].bias, 0)
            nn.init.constant_(self.local_head.final.ada[-1].weight, 0)
            nn.init.constant_(self.local_head.final.ada[-1].bias, 0)
            nn.init.constant_(self.local_head.final.head.weight, 0)
            nn.init.constant_(self.local_head.final.head.bias, 0)

    # -- bookkeeping --------------------------------------------------------

    def param_breakdown(self):
        def count(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        per_block = count(self.blocks[0]) if len(self.blocks) else 0
        return {
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'patch_embed': count(self.patch_embed),
            'backbone_blocks': count(self.blocks),
            'per_block': per_block,
            'local_head': count(self.local_head),
            'final_head': count(self.final),
            'conditioning': count(self.time_mlp) + count(self.cond_proj)
                            + (count(self.class_emb) + count(self.class_mlp)
                               if self.class_conditional else 0),
        }


def depth_for_target_params(target_params, model_kwargs):
    """Depth whose parameter count lands closest to `target_params`.

    Used to keep the autoregressive baseline parameter-matched to the diffusion
    models rather than eyeballing the depth.
    """
    kwargs = dict(model_kwargs)
    kwargs['depth'] = 2
    probe = ARDiT3D(**kwargs)
    br = probe.param_breakdown()
    per_block = br['per_block']
    fixed = br['total'] - 2 * per_block
    del probe
    depth = max(1, round((target_params - fixed) / max(1, per_block)))
    return depth, fixed + depth * per_block


def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0,
                       allowed_block_mask=None):
    """Sample block indices from logits of shape [..., V]."""
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    if allowed_block_mask is not None:
        mask = allowed_block_mask.to(logits.device).bool()
        if mask.dim() == 1:
            mask = mask.view(*([1] * (logits.dim() - 1)), -1)
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(-2)
        logits = logits.masked_fill(~mask, float('-inf'))

    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / float(temperature)

    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = (cumulative - probs) > float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    probs = torch.where((row_sums <= 0) | ~torch.isfinite(row_sums), uniform,
                        probs / torch.clamp(row_sums, min=1e-8))

    shape = probs.shape[:-1]
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).reshape(shape)


if __name__ == '__main__':
    torch.manual_seed(0)

    # layout round-trip
    x = torch.randint(0, 7, (2, 8, 8, 8))
    for p in (1, 2, 4):
        assert torch.equal(patch_voxels_to_volume(volume_to_patch_voxels(x, p), p), x), p
    print('layout round-trip ok')

    model = ARDiT3D(vocab_size=11, hidden_channels=96, image_size=8, patch_size=2,
                    depth=2, num_heads=4, num_classes=3, local_head_depth=1,
                    local_head_dim=48, local_head_heads=4).eval()

    # give the zero-initialised heads something to say
    with torch.no_grad():
        model.local_head.final.head.weight.normal_(std=0.05)
        for blk in model.local_head.blocks:
            blk.ada[-1].weight.normal_(std=0.05)
        for blk in model.blocks:
            blk.ada[-1].weight.normal_(std=0.05)

    vol = torch.randint(0, 11, (2, 8, 8, 8))
    labels = torch.randint(0, 3, (2,))
    logits, targets = model(vol, labels, cond_drop_prob=0.)
    assert torch.equal(model.volume_from_targets(targets), vol)
    print('teacher-forced logits', tuple(logits.shape), 'targets', tuple(targets.shape))

    # causality: changing voxel k must leave the predictions for voxels <= k
    # untouched and move at least one later prediction.
    flat = model.targets_from_volume(vol).clone()
    flat[:, 0, 1] = (flat[:, 0, 1] + 1) % 11
    logits2, _ = model(model.volume_from_targets(flat), labels, cond_drop_prob=0.)
    delta = (logits - logits2).abs()
    assert delta[:, 0, :2].max() < 1e-4, delta[:, 0, :2].max()
    assert delta[:, 0, 2:].max() > 1e-4, 'later predictions did not react'
    print('causality ok (drift before %.2e, after %.2e)'
          % (delta[:, 0, :2].max().item(), delta.max().item()))

    # the cached incremental path must reproduce the parallel path exactly:
    # greedily generated volumes are a fixed point of teacher forcing.
    greedy = model.generate(2, labels, cond_scale=1.0, temperature=0.0, progress=False)
    gl, gt = model(greedy, labels, cond_drop_prob=0.)
    assert torch.equal(gl.argmax(-1), gt), 'incremental and parallel paths disagree'
    print('kv-cache / teacher-forcing equivalence ok')

    forced = model.generate(2, labels, temperature=1.0, progress=False,
                            known_volume=vol, known_mask=torch.ones_like(vol, dtype=torch.bool))
    assert torch.equal(forced, vol), 'hard constraints were not respected'
    print('hard block constraints ok')

    print('params', model.param_breakdown())
