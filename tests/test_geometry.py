"""Unit tests for the image/mask geometry of LanPaintInpaintPipeline._preprocess.

No model: a stub adapter carries the real VaeImageProcessor. The mask is checked
against an independent nearest-neighbour oracle and the image against PIL's own
bicubic resize, so the two branches are tested separately, each with its filter.

Run with: pytest tests/test_geometry.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from lanpaint_pipeline.pipeline import LanPaintInpaintPipeline


def make_pipeline(vae_scale_factor=16):
    pipe = LanPaintInpaintPipeline.__new__(LanPaintInpaintPipeline)
    pipe.adapter = SimpleNamespace(
        dtype=torch.float32,
        vae_scale_factor=vae_scale_factor,
        image_processor=VaeImageProcessor(vae_scale_factor=vae_scale_factor),
        supports_alpha=False,
    )
    return pipe


def preprocess(image, mask, height=None, width=None):
    return make_pipeline()._preprocess(image, mask, None, height, width, torch.device("cpu"))


def nearest_oracle(source, size):
    """Sample the source pixel under each output pixel centre."""
    w, h = size
    src_h, src_w = source.shape
    rows = np.minimum(((np.arange(h) + 0.5) * src_h / h).astype(int), src_h - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * src_w / w).astype(int), src_w - 1)
    return source[rows][:, cols]


def rectangles(width, height, seed=0):
    """A keep-mask (255 = keep) with a few random edit rectangles at integer coordinates."""
    rng = np.random.default_rng(seed)
    keep = np.full((height, width), 255, np.uint8)
    for _ in range(4):
        x0, y0 = rng.integers(0, width - 8), rng.integers(0, height - 8)
        keep[y0:y0 + rng.integers(1, 40), x0:x0 + rng.integers(1, 40)] = 0
    return keep


@pytest.mark.parametrize("size, explicit, final", [
    ((96, 80), None, (96, 64)),          # aligned down, no explicit size
    ((70, 50), (64, 64), (64, 64)),      # explicit size, already aligned
    ((300, 200), (90, 100), (64, 96)),   # explicit size, aligned down
    ((1500, 900), None, (1312, 768)),    # capped at 1024² pixels, then aligned
])
def test_mask_follows_a_nearest_oracle_and_image_a_bicubic_resize(size, explicit, final):
    rng = np.random.default_rng(1)
    source = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    keep = rectangles(*size)
    height, width = (explicit[1], explicit[0]) if explicit else (None, None)
    orig, mask_keep, img_tensor, ih, iw = preprocess(Image.fromarray(source), Image.fromarray(keep), height, width)

    assert (iw, ih) == final and orig.size == final and tuple(img_tensor.shape[-2:]) == (ih, iw)
    expected_mask = nearest_oracle(keep, final) > 127
    assert np.array_equal(mask_keep[0, 0].numpy() > 0.5, expected_mask)
    assert np.array_equal(np.asarray(orig), np.asarray(Image.fromarray(source).resize(final, Image.BICUBIC)))


def test_a_stripe_stays_under_its_mask_on_an_unaligned_image():
    # 96x80 is aligned to 96x64: the main branch cropped the mask but resized the image.
    image = np.zeros((80, 96, 3), np.uint8)
    image[16:32, :, 0] = 255
    edit = np.full((80, 96), 255, np.uint8)
    edit[16:32, :] = 0
    orig, mask_keep, _, ih, iw = preprocess(Image.fromarray(image), Image.fromarray(edit))
    stripe = np.flatnonzero((np.asarray(orig)[..., 0] > 127).any(axis=1))
    masked = np.flatnonzero((mask_keep[0, 0].numpy() < 0.5).any(axis=1))
    assert (iw, ih) == (96, 64)
    assert abs(stripe[0] - masked[0]) <= 1 and abs(stripe[-1] - masked[-1]) <= 1


def test_rgba_mask_alpha_follows_the_same_mapping():
    keep = rectangles(96, 80, seed=2)
    rgba = np.zeros((80, 96, 4), np.uint8)
    rgba[..., 3] = keep
    _, mask_keep, _, ih, iw = preprocess(Image.new("RGB", (96, 80)), Image.fromarray(rgba, "RGBA"))
    assert np.array_equal(mask_keep[0, 0].numpy() > 0.5, nearest_oracle(keep, (iw, ih)) > 127)


@pytest.mark.parametrize("mask_size", [(48, 40), (192, 160)])
def test_a_mask_of_another_size_is_resized_to_the_run_size(mask_size):
    keep = rectangles(*mask_size, seed=3)
    _, mask_keep, _, ih, iw = preprocess(Image.new("RGB", (96, 80)), Image.fromarray(keep))
    assert (iw, ih) == (96, 64)
    assert np.array_equal(mask_keep[0, 0].numpy() > 0.5, nearest_oracle(keep, (iw, ih)) > 127)


@pytest.mark.parametrize("height, width", [(64, None), (None, 64)])
def test_a_single_dimension_falls_back_to_the_image_size(height, width):
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (80, 96, 3), dtype=np.uint8))
    keep = Image.fromarray(rectangles(96, 80, seed=4))
    _, single_mask, single_image, ih, iw = preprocess(image, keep, height, width)
    _, mask, image_tensor, *size = preprocess(image, keep)
    assert (iw, ih) == (size[1], size[0]) == (96, 64)
    assert torch.equal(single_mask, mask) and torch.equal(single_image, image_tensor)


@pytest.mark.parametrize("image_size, height, width", [((96, 80), 16, 64), ((96, 80), 0, 64), ((20, 20), None, None)])
def test_an_empty_aligned_size_is_refused_before_the_model(image_size, height, width):
    with pytest.raises(ValueError, match="empty"):
        preprocess(Image.new("RGB", image_size), Image.new("L", image_size, 255), height, width)
