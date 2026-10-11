"""Unit tests for the opt-in alpha path of LanPaintInpaintPipeline.

No model: stub adapters carry the real VaeImageProcessor, one declaring
supports_alpha (as Qwen-Image 2.1 does) and one not (every RGB model). The
composite is checked against the straight-alpha blend equation per channel.

Run with: pytest tests/test_alpha.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from scipy.ndimage import correlate, maximum_filter

from lanpaint_pipeline.pipeline import LanPaintInpaintPipeline
from lanpaint_pipeline.utils import blend_with_smooth_mask


def make_pipeline(supports_alpha):
    pipe = LanPaintInpaintPipeline.__new__(LanPaintInpaintPipeline)
    pipe.adapter = SimpleNamespace(
        dtype=torch.float32,
        vae_scale_factor=16,
        image_processor=VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64),
        supports_alpha=supports_alpha,
    )
    return pipe


def preprocess(supports_alpha, image, mask=None, outpaint=None):
    return make_pipeline(supports_alpha)._preprocess(image, mask, outpaint, None, None, torch.device("cpu"))


def canvas(alpha, size=(64, 64)):
    """Same purple RGB everywhere (the colour hidden under Example_32's transparency), one alpha."""
    rgba = np.zeros((size[1], size[0], 4), np.uint8)
    rgba[..., :3] = (164, 58, 207)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


KEEP = Image.new("L", (64, 64), 255)


@pytest.mark.parametrize("as_path", [False, True])
def test_alpha_reaches_the_vae_input(tmp_path, as_path):
    tensors = []
    for alpha in (0, 128, 255):
        image = canvas(alpha)
        if as_path:
            image.save(tmp_path / f"canvas-{alpha}.png")
            image = str(tmp_path / f"canvas-{alpha}.png")
        orig, _, tensor, _, _ = preprocess(True, image, KEEP)
        assert orig.mode == "RGBA" and tensor.shape == (1, 4, 64, 64)
        # the 8-bit alpha is recovered exactly from the [-1, 1] VAE input
        assert torch.equal(((tensor[0, 3] + 1) / 2 * 255).round(), torch.full((64, 64), float(alpha)))
        tensors.append(tensor)
    assert all(torch.equal(t[:, :3], tensors[0][:, :3]) for t in tensors)
    assert not torch.equal(tensors[0], tensors[1]) and not torch.equal(tensors[1], tensors[2])


def test_an_rgb_canvas_becomes_opaque():
    _, _, tensor, _, _ = preprocess(True, Image.new("RGB", (64, 64), (10, 20, 30)), KEEP)
    assert torch.equal(tensor[0, 3], torch.ones(64, 64))


def test_rgb_models_still_get_rgb():
    # The RGB branch is the original line: transparency is dropped, not composited.
    orig, _, tensor, _, _ = preprocess(False, canvas(0), KEEP)
    _, _, reference, _, _ = preprocess(False, canvas(0).convert("RGB"), KEEP)
    assert orig.mode == "RGB" and tensor.shape == (1, 3, 64, 64) and torch.equal(tensor, reference)


def test_outpainting_is_refused_with_alpha():
    with pytest.raises(ValueError, match="Outpainting"):
        preprocess(True, canvas(255), outpaint="r24")


def test_rgb_outpainting_is_unchanged():
    # 40x40 image padded 24 px on the right: a 64x40 canvas, extended down to 64x64 in grey.
    orig, keep, tensor, ih, iw = preprocess(False, Image.new("RGB", (40, 40), (200, 10, 10)), outpaint="r24")
    expected = np.full((64, 64, 3), 127, np.uint8)
    expected[:40, :40] = (200, 10, 10)
    expected_keep = np.zeros((64, 64), np.float32)
    expected_keep[:40, :20] = 1.0  # the 20 px next to the new border are redrawn
    assert (iw, ih) == (64, 64)
    assert np.array_equal(np.asarray(orig), expected)
    assert np.array_equal(keep[0, 0].numpy(), expected_keep)
    reference = VaeImageProcessor(vae_scale_factor=16).preprocess(Image.fromarray(expected), height=64, width=64)
    assert torch.equal(tensor, reference)


def blend_oracle(orig, gen, keep, overlap=9):
    """Float64 reference in 0-255 units: the keep mask dilated by overlap // 2, blurred by a
    normalized Gaussian (sigma (overlap - 1) / 4, zero padding), then orig * m + gen * (1 - m)
    for every channel, alpha included (straight alpha)."""
    radius = overlap // 2
    m = maximum_filter(keep.astype(np.float64), size=overlap, mode="constant", cval=0.0)
    axis = np.arange(overlap) - radius
    g = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * ((overlap - 1) / 4) ** 2))
    m = np.clip(correlate(m, g / g.sum(), mode="constant", cval=0.0), 0.0, 1.0)[..., None]
    return orig * m + gen * (1.0 - m)


def test_rgba_composite_follows_the_straight_alpha_equation():
    rng = np.random.default_rng(3)
    orig = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
    gen = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
    orig[..., 3] = rng.choice([0, 128, 255], (96, 96))  # hidden RGB differs under each alpha
    keep = np.ones((96, 96), np.float32)
    keep[32:64, 32:64] = 0
    out = np.asarray(blend_with_smooth_mask(
        Image.fromarray(orig, "RGBA"), Image.fromarray(gen, "RGBA"), torch.from_numpy(keep)[None, None], overlap=9))

    assert out.shape == (96, 96, 4)
    yy, xx = np.mgrid[:96, :96]
    away_from_border = np.minimum.reduce([yy, xx, 95 - yy, 95 - xx]) >= 4
    kept = (keep == 1) & away_from_border
    assert np.array_equal(out[kept], orig[kept])  # all four channels, hidden RGB included
    deep_edit = np.zeros_like(keep, bool)
    deep_edit[42:54, 42:54] = True  # beyond the 4 px dilation and the 9x9 blur
    assert np.array_equal(out[deep_edit], gen[deep_edit])
    # Everywhere, transition and canvas border included: the 8-bit output is the float64
    # equation truncated, as the composite converts with astype(uint8).
    reference = blend_oracle(orig.astype(np.float64), gen.astype(np.float64), keep)
    gap = reference - out
    assert ((gap > -1e-3) & (gap < 1 + 1e-3)).all()
    transition = (gap.max(axis=-1) > 0) & ~kept & ~deep_edit
    assert transition.sum() > 500  # the band really mixes both images
