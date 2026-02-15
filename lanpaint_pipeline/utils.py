"""
Shared utilities for LanPaint inpainting pipeline.

- Notation helpers: flow <-> VP <-> VE conversions
- Gaussian kernel & smooth mask blend
- Image loading (alpha-preserving for masks)
"""

import io
from typing import Union

import numpy as np
import torch
from PIL import Image


def flow_to_abt(flow_t_val: float) -> float:
    """Flow time t -> alpha_bar_t (VP schedule parameter)."""
    t = float(min(max(flow_t_val, 1e-3), 1.0 - 1e-3))
    return (1.0 - t) ** 2 / ((1.0 - t) ** 2 + t ** 2 + 1e-8)


def flow_to_ve_sigma(flow_t_val: float) -> float:
    """Flow time t -> VE sigma."""
    t = float(min(max(flow_t_val, 1e-6), 1.0 - 1e-2))
    return t / (1.0 - t)


def make_current_times(flow_t_val: float, device: torch.device):
    """
    Build LanPaint's ``current_times`` tuple: (VE_sigma, alpha_bar_t, flow_t).
    Each element has shape ``(1,)``.
    """
    return (
        torch.tensor([flow_to_ve_sigma(flow_t_val)], device=device, dtype=torch.float32),
        torch.tensor([flow_to_abt(flow_t_val)], device=device, dtype=torch.float32),
        torch.tensor([flow_t_val], device=device, dtype=torch.float32),
    )


def gaussian_kernel_2d(kernel_size: int, device=None) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel (matches LanPaint's reference)."""
    sigma = (kernel_size - 1) / 4.0
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    xg, yg = torch.meshgrid(x, y, indexing="ij")
    k = torch.exp(-(xg ** 2 + yg ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blend_with_smooth_mask(
    orig_pil: Image.Image,
    gen_pil: Image.Image,
    mask_keep: torch.Tensor,
    overlap: int = 9,
    device=None,
) -> Image.Image:
    """
    Smooth pixel-space blend: orig * smoothed_mask + generated * (1 - smoothed_mask).

    Parameters
    ----------
    orig_pil : PIL.Image
        Original (unedited) image.
    gen_pil : PIL.Image
        Generated (inpainted) image.
    mask_keep : Tensor (1, 1, H, W)
        Pixel-space keep mask. 1 = keep original, 0 = use generated.
    overlap : int
        Gaussian kernel size for smoothing the mask boundary. Set < 3 for hard composite.
    device : torch.device, optional
    """
    m_keep = mask_keep.float()

    if overlap < 3:
        # Hard composite
        mask_pil = Image.fromarray(
            (m_keep[0, 0].cpu().numpy() * 255).astype(np.uint8)
        ).resize(gen_pil.size, Image.NEAREST)
        return Image.composite(
            orig_pil.resize(gen_pil.size, Image.BICUBIC), gen_pil, mask_pil
        )

    if overlap % 2 == 0:
        overlap += 1

    w, h = gen_pil.size
    orig_t = (
        torch.from_numpy(np.array(orig_pil.resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    gen_t = (
        torch.from_numpy(np.array(gen_pil).astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    m = torch.nn.functional.interpolate(m_keep, size=(h, w), mode="nearest").squeeze(1)
    m = torch.nn.functional.max_pool2d(m, kernel_size=overlap, stride=1, padding=overlap // 2)
    kern = gaussian_kernel_2d(overlap, device=m.device).view(1, 1, overlap, overlap)
    m = (
        torch.nn.functional.conv2d(m.unsqueeze(1), kern, padding=overlap // 2)
        .squeeze(1)
        .clamp(0, 1)
    )

    out = orig_t * m.unsqueeze(1) + gen_t * (1.0 - m.unsqueeze(1))
    return Image.fromarray(
        (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )


def load_image_preserve_alpha(source: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image while preserving the alpha channel.

    IMPORTANT: ``diffusers.utils.load_image`` converts to RGB by default,
    dropping the alpha channel. For mask images where the alpha channel
    carries mask information, use this function instead.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            import requests

            resp = requests.get(source)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).copy()
        return Image.open(source)
    raise TypeError(f"Unsupported image source type: {type(source)}")
