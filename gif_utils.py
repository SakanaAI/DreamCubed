"""
GIF visualization utilities for the MD4 discrete diffusion denoising process.

Renders animated GIFs showing how a 3D Minecraft voxel chunk is progressively
denoised.  The default mode treats masked voxels as air (invisible), so blocks
appear to pop into existence as they are unmasked during the reverse process.

Rendering is delegated to :class:`MinecraftVisualizerPyVista` (textured or
flat-colour), keeping this module focused on the diffusion-to-GIF pipeline.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from visualization_utils import MinecraftVisualizerPyVista


# ---------------------------------------------------------------------------
# Intermediate-step helpers
# ---------------------------------------------------------------------------

def subsample_steps(
    intermediates: torch.Tensor,
    num_frames: int,
    *,
    sample_index: int = 0,
) -> torch.Tensor:
    """
    Evenly subsample a sequence of intermediate diffusion states.

    Args:
        intermediates: ``[T, H, W, D]`` or ``[B, T, H, W, D]`` model indices.
        num_frames: Desired number of output frames.
        sample_index: If batched (5-D), which sample to extract.

    Returns:
        ``[min(num_frames, T), H, W, D]`` tensor.
    """
    if intermediates.dim() == 5:
        intermediates = intermediates[sample_index]
    if intermediates.dim() != 4:
        raise ValueError(
            f"Expected [T,H,W,D] or [B,T,H,W,D], got shape {tuple(intermediates.shape)}"
        )
    T = intermediates.shape[0]
    if num_frames >= T:
        return intermediates
    idx = torch.linspace(0, T - 1, steps=num_frames).round().long()
    idx[-1] = T - 1  # always include the final step
    return intermediates[idx]


def _build_block_id_lut(converter) -> torch.Tensor:
    """Vectorised lookup table mapping model indices to original block IDs."""
    index_keys = list(getattr(converter, "index_to_block", {}).keys())
    if not index_keys:
        return torch.arange(256, dtype=torch.long)
    K = int(max(index_keys)) + 1
    lut = torch.zeros(K, dtype=torch.long)
    for idx, bid in converter.index_to_block.items():
        lut[int(idx)] = int(bid)
    return lut


# ---------------------------------------------------------------------------
# Core GIF renderer
# ---------------------------------------------------------------------------

def render_diffusion_gif(
    intermediates: torch.Tensor,
    out_path: str,
    converter,
    *,
    num_frames: int = 30,
    image_size: int = 512,
    fps: int = 10,
    textured: bool = True,
    zoom: float = 1.0,
    sample_index: int = 0,
    hold_final_frames: int = 4,
    include_initial_empty: bool = True,
    show_axis: bool = False,
    camera_position=None,
    optimize_gif: bool = False,
    textures_dir: Optional[str] = None,
) -> str:
    """
    Render an animated GIF of the MD4 denoising process.

    Masked voxels (token ``num_classes``) are rendered as air, so the chunk
    appears to materialise from nothing as blocks are progressively revealed.

    Args:
        intermediates: ``[T, H, W, D]`` or ``[B, T, H, W, D]`` tensor of
            model-index values at successive denoising steps.
        out_path: Destination ``.gif`` path.
        converter: :class:`BlockBiomeConverter` used to map model indices to
            the original Minecraft block IDs needed by the visualiser.
        num_frames: How many intermediate frames to render (the full sequence
            is evenly subsampled to this count).
        image_size: Frame resolution in pixels (square).
        fps: Frames per second in the output GIF.
        textured: ``True`` for textured voxel rendering, ``False`` for flat
            colour cubes.
        zoom: Camera zoom factor applied after default framing.
        sample_index: Which sample to use from a batched intermediates tensor.
        hold_final_frames: Extra copies of the final fully-denoised frame
            appended so the result lingers before looping.
        include_initial_empty: Prepend an all-air (empty) frame.
        show_axis: Render coordinate axes.
        camera_position: Fixed PyVista camera position (3-tuple of tuples:
            ``(position, focal_point, viewup)``).  When *None* the camera is
            set automatically on the first frame and reused for consistency.
        optimize_gif: Pillow ``optimize`` flag (slower but smaller files).
        textures_dir: Optional texture directory for textured rendering.

    Returns:
        The *out_path* string.
    """
    # ------------------------------------------------------------------
    # 1. Subsample intermediates to the requested frame budget
    # ------------------------------------------------------------------
    steps = subsample_steps(intermediates, num_frames, sample_index=sample_index)
    steps = steps.detach().cpu().long()
    T, H, W, D = steps.shape

    # The mask token sits one past the last valid model index.
    mask_token_id = int(steps.max().item())

    try:
        air_index = int(converter.get_air_block_index())
    except Exception:
        air_index = 0

    lut = _build_block_id_lut(converter)

    # ------------------------------------------------------------------
    # 2. Prepare the visualiser
    # ------------------------------------------------------------------
    if textured:
        if textures_dir and os.path.exists(textures_dir):
            visualizer = MinecraftVisualizerPyVista(
                textures_dir=textures_dir,
                build_textures=True,
            )
        else:
            visualizer = MinecraftVisualizerPyVista(build_textures=True)
        render_fn = visualizer.visualize_chunk_textured
    else:
        visualizer = MinecraftVisualizerPyVista()
        render_fn = visualizer.visualize_chunk

    frames: List[Image.Image] = []
    plotter = None
    saved_camera = None

    def _render_frame(model_indices: torch.Tensor):
        nonlocal plotter, saved_camera

        clean = model_indices.clone()
        # Any index outside the LUT range (i.e. the mask token) becomes air.
        clean[clean >= lut.shape[0]] = air_index
        block_ids = lut[clean]

        if plotter is not None:
            try:
                plotter.clear()
            except Exception:
                pass

        plotter = render_fn(
            block_ids, plotter=plotter, interactive=False, show_axis=show_axis,
        )

        # Lock the camera to the position established on the first frame so
        # the viewpoint doesn't drift as geometry changes between frames.
        if saved_camera is not None:
            plotter.camera_position = saved_camera
        elif camera_position is not None:
            plotter.camera_position = camera_position

        if zoom != 1.0 and hasattr(plotter, "camera"):
            plotter.camera.zoom(float(zoom))

        img = plotter.screenshot(
            window_size=(image_size, image_size),
            transparent_background=False,
            return_img=True,
        )

        if saved_camera is None:
            saved_camera = plotter.camera_position

        frames.append(Image.fromarray(img))

    # ------------------------------------------------------------------
    # 3. Render frames
    # ------------------------------------------------------------------
    if include_initial_empty:
        empty = torch.full((H, W, D), fill_value=air_index, dtype=torch.long)
        _render_frame(empty)

    for t in range(T):
        _render_frame(steps[t])

    if hold_final_frames > 0 and frames:
        for _ in range(hold_final_frames):
            frames.append(frames[-1].copy())

    # ------------------------------------------------------------------
    # 4. Assemble and save the GIF
    # ------------------------------------------------------------------
    if plotter is not None:
        try:
            plotter.close()
        except Exception:
            pass

    out_dir = os.path.dirname(str(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    duration_ms = max(1, int(1000 / max(1, fps)))
    save_kw = dict(
        save_all=True, append_images=frames[1:], duration=duration_ms, loop=0,
    )
    if optimize_gif:
        save_kw["optimize"] = True

    frames[0].save(str(out_path), **save_kw)
    return out_path
