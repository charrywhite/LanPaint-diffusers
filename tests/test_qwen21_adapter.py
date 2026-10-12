"""Unit tests for the Qwen-Image 2.1 adapter.

No model weights: a fake pipe borrows QwenImage21Pipeline's own prepare_latents,
VAE normalisation and pack/unpack, runs a tiny real QwenImage21Transformer2DModel
(FP32 on CPU), and stands in for the VAE and the VL encoder with stubs whose
outputs depend on the pixels. The bf16 rounding of the real pipeline is out of
reach of a float32 test; only the timestep expression is checked for it here.

Run with: pytest tests/test_qwen21_adapter.py
"""

import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from PIL import Image

from lanpaint_pipeline.adapters.qwen21 import (
    Qwen21Adapter,
    append_target_slots,
    check_condition_layout,
    guided_x0,
    model_timestep,
    project_keep_mask,
    schedule_sigmas,
)
from lanpaint_pipeline.pipeline import LanPaintConfig, LanPaintInpaintPipeline

# scheduler_config.json of Qwen/Qwen-Image-2.1
SCHEDULER = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9, shift=1.0,
                 shift_terminal=0.02, time_shift_type="exponential", use_dynamic_shifting=True)


class StubVAE:
    """16x RGBA VAE: each latent cell holds its tile's mean RGBA, tiled over the 64 channels."""

    dtype = torch.float32
    config = SimpleNamespace(z_dim=64, latents_mean=[0.0] * 64, latents_std=[1.0] * 64)

    def __init__(self):
        self.inputs = []

    def encode(self, x):
        self.inputs.append(x.clone())
        tiles = torch.nn.functional.avg_pool2d(x[:, :, 0], 16)
        return SimpleNamespace(latents=tiles.repeat(1, 16, 1, 1).unsqueeze(2))

    def decode(self, z, return_dict=False):
        return (torch.nn.functional.interpolate(z[:, :4, 0], scale_factor=16, mode="nearest").unsqueeze(2),)


def stub_encode_prompt(pipe, prompt, image, device):
    """VL stand-in with Qwen's layout: two text rows, then per image its slots and a separator.
    Slot values follow the image's mean pixel, so a different image gives different embeds."""
    pipe.shown.append((prompt, [im.copy() for im in image]))
    rows, pad = [torch.full((2, 8), float(len(prompt)))], [False, False]
    for im in image:
        slots = (im.width // 32) * (im.height // 32)
        rows += [torch.full((slots, 8), float(np.asarray(im).mean()) / 255), torch.zeros(1, 8)]
        pad += [True] * slots + [False]
    return torch.cat(rows)[None], None, torch.tensor([pad])


def tiny_transformer(seed=0):
    torch.manual_seed(seed)
    return QwenImage21Transformer2DModel(num_layers=2, num_attention_heads=1, attention_head_dim=32,
                                         context_in_dim=8, axes_dims_rope=(8, 12, 12), mlp_ratio=2).eval()


def make_adapter(transformer=None):
    pipe = SimpleNamespace(vae=StubVAE(), vae_scale_factor=16, latent_channels=64, shown=[],
                           image_processor=VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64),
                           scheduler=FlowMatchEulerDiscreteScheduler(**SCHEDULER),
                           config=SimpleNamespace(sample_sigmas=None), _execution_device=torch.device("cpu"),
                           transformer=transformer or tiny_transformer())
    for name in ("prepare_latents", "_encode_vae_image"):
        setattr(pipe, name, MethodType(getattr(QwenImage21Pipeline, name), pipe))
    pipe._pack_latents = QwenImage21Pipeline._pack_latents
    pipe._unpack_latents = QwenImage21Pipeline._unpack_latents
    pipe.encode_prompt = MethodType(stub_encode_prompt, pipe)
    adapter = Qwen21Adapter(pipe)
    adapter.condition_resolution = 64  # keep conditions at 4x4 latents on CPU
    return adapter


def rgba(colour, size=(64, 64), alpha=255):
    return Image.new("RGBA", size, (*colour, alpha))


def canvas_tensor(image):
    return VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).preprocess(image)


def prepare(adapter, image, prompt="rebuild the earcups", negative=""):
    adapter.encode_prompt(prompt, negative, torch.device("cpu"))
    with torch.no_grad():
        return adapter.encode_and_prepare(canvas_tensor(image), image.height, image.width, None, torch.device("cpu"))


# ---- pure helpers ----


def test_registry_entry():
    from lanpaint_pipeline.registry import get_adapter_class, get_model_spec
    spec = get_model_spec("qwen-image-2.1")
    assert spec.pipeline_cls_path == "diffusers.QwenImage21Pipeline" and spec.requires_ref_at_inference
    assert get_adapter_class("qwen-image-2.1") is Qwen21Adapter


def test_keep_mask_is_sampled_at_cell_centres_like_comfyui():
    keep = torch.ones(1, 1, 64, 64)
    keep[..., 8, :] = 0   # row 8 is sampled for latent row 0 (centre of 0..15)
    keep[..., 30, :] = 0  # row 30 falls between samples (24 and 40): the stroke disappears
    cells = project_keep_mask(keep, 4, 4).view(4, 4)
    assert cells[0].eq(0).all() and cells[1:].eq(1).all()
    assert project_keep_mask(torch.zeros(1, 1, 64, 64), 4, 4).eq(0).all()


def test_keep_mask_must_be_one_single_channel_mask():
    for shape in ((2, 1, 64, 64), (1, 3, 64, 64), (64, 64)):
        with pytest.raises(ValueError, match="keep-mask"):
            project_keep_mask(torch.ones(shape), 4, 4)


def test_schedule_follows_explicit_then_checkpoint_then_linear():
    np.testing.assert_allclose(schedule_sigmas(4), [1, 0.75, 0.5, 0.25])
    np.testing.assert_array_equal(schedule_sigmas(40, checkpoint=[1.0, 0.5]), [1.0, 0.5])
    np.testing.assert_array_equal(schedule_sigmas(40, [1.0, 0.25], [1.0, 0.5]), [1.0, 0.25])
    for bad in ([], [1.0, 0.0], [0.5, 0.75], [1.5], [float("nan")]):
        with pytest.raises(ValueError, match="sigmas"):
            schedule_sigmas(4, bad)
    for bad in (0, -3, True, 2.5):
        with pytest.raises(ValueError, match="num_steps"):
            schedule_sigmas(bad)


def test_timestep_rounds_like_the_pipeline_in_bf16():
    scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
    scheduler.set_timesteps(sigmas=np.linspace(1, 1 / 20, 20), mu=0.6935)
    for t, sigma in zip(scheduler.timesteps, scheduler.sigmas[:-1]):
        expected = t.expand(1).to(torch.bfloat16) / 1000  # QwenImage21Pipeline's expression
        assert torch.equal(model_timestep(sigma.item(), torch.bfloat16), expected)


def test_guidance_matches_the_pipeline_formula():
    x, vc, vu = torch.randn(1, 16, 64), torch.randn(1, 16, 64), torch.randn(1, 16, 64)
    main, big = guided_x0(x, 0.7, vc, vu, 4.0, -0.5)
    assert torch.equal(main, x - 0.7 * (vu + 4.0 * (vc - vu)))
    assert torch.equal(big, x - 0.7 * (vu + -0.5 * (vc - vu)))
    ref = x.double() - 0.7 * (vu.double() + 4.0 * (vc.double() - vu.double()))
    assert (main.double() - ref).abs().max() < 1e-5
    assert torch.equal(guided_x0(x, 0.7, vc, vu, 1.0, 1.0)[0], x - 0.7 * vc)  # main CFG off at <= 1
    one, same = guided_x0(x, 0.7, vc, None, 1.0, 1.0)
    assert one is same and torch.equal(one, x - 0.7 * vc)
    with pytest.raises(ValueError, match="unconditional"):
        guided_x0(x, 0.7, vc, None, 4.0, 1.0)
    # bf16 velocities are combined in float32, not in bf16
    c = torch.linspace(-1, 1, 32).reshape(1, 4, 8).bfloat16()
    u = (c * 0.137).bfloat16()
    ones = torch.ones(1, 4, 8)
    expected = ones - 0.7 * (u.float() + 4.0 * (c.float() - u.float()))
    assert torch.equal(guided_x0(ones, 0.7, c, u, 4.0, 1.0)[0], expected)


def test_condition_layout_is_one_run_per_condition():
    pad = torch.tensor([[False, True, True, True, True, False, True, False]])
    check_condition_layout(pad, [16, 4])
    with pytest.raises(ValueError, match="slot runs"):
        check_condition_layout(pad, [16, 16])
    assert append_target_slots(pad, 16).shape == (1, 12) and append_target_slots(pad, 16)[0, -4:].all()
    with pytest.raises(ValueError, match="boolean"):
        check_condition_layout(pad.int(), [16, 4])
    with pytest.raises(ValueError, match="boolean"):
        check_condition_layout(pad.repeat(2, 1), [16, 4])
    with pytest.raises(ValueError, match="multiples of 4"):
        check_condition_layout(pad, [16, 6])
    with pytest.raises(ValueError, match="multiple of 4"):
        append_target_slots(pad, 18)


# ---- conditions ----


def test_canvas_is_image1_and_references_follow_with_their_own_size():
    adapter = make_adapter()
    adapter.set_references([rgba((10, 200, 30), size=(128, 64))])
    prepare(adapter, rgba((164, 58, 207)))
    _, shown = adapter.pipe.shown[-1]
    assert [im.mode for im in shown] == ["RGBA", "RGBA"]  # alpha reaches encode_prompt, which flattens it
    assert [im.size for im in shown] == [(64, 64), (96, 32)]  # calculate_dimensions(64², ratio)
    assert adapter._img_shapes == [[(1, 4, 4), (1, 2, 6), (1, 4, 4)]]
    assert adapter._condition_latents.shape == (1, 16 + 12, 64)


def test_vl_encoder_and_vae_see_the_same_resized_pixels():
    adapter = make_adapter()
    gradient = np.linspace(0, 255, 100 * 37 * 4).reshape(37, 100, 4).astype(np.uint8)  # asymmetric
    adapter.set_references([Image.fromarray(gradient)])
    prepare(adapter, rgba((164, 58, 207)))
    shown = adapter.pipe.shown[-1][1]
    vae_inputs = adapter.pipe.vae.inputs[:len(shown)]  # the conditions, before the canvas at run size
    for image, vae_input in zip(shown, vae_inputs):
        pixels = ((vae_input[0, :, 0] + 1) / 2 * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)
        assert np.array_equal(pixels, np.asarray(image))


def test_canvas_alpha_changes_what_is_kept_and_what_is_shown():
    opaque, clear = make_adapter(), make_adapter()
    kept_opaque = prepare(opaque, rgba((164, 58, 207), alpha=255)).latent
    kept_clear = prepare(clear, rgba((164, 58, 207), alpha=0)).latent
    assert not torch.equal(kept_opaque, kept_clear)
    assert not torch.equal(opaque._condition_latents, clear._condition_latents)
    assert clear.pipe.shown[-1][1][0].getpixel((0, 0))[3] == 0


def test_references_are_replaced_and_cleared_explicitly():
    adapter = make_adapter()
    adapter.set_references([rgba((1, 2, 3)), rgba((4, 5, 6))])
    assert len(adapter.references) == 2
    adapter.set_references([])
    prepare(adapter, rgba((164, 58, 207)))
    assert len(adapter.pipe.shown[-1][1]) == 1
    with pytest.raises(TypeError):
        adapter.set_references(["material.png"])


def test_negative_is_encoded_lazily_with_the_same_conditions():
    adapter = make_adapter()
    adapter.set_references([rgba((10, 200, 30))])
    prepare(adapter, rgba((164, 58, 207)))
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        adapter.predict_x0(x, 0.8, 1.0, 1.0)
        assert len(adapter.pipe.shown) == 1  # no guidance: no negative branch
        adapter.predict_x0(x, 0.8, 4.0, 1.0)
    (prompt, positive), (negative_prompt, negative) = adapter.pipe.shown
    assert negative_prompt == "" and len(negative) == len(positive) == 2
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(positive, negative))


def test_text_prompt_replaces_injected_conditioning():
    adapter = make_adapter()
    prepare(adapter, rgba((164, 58, 207)))
    embeds, _, pad = adapter._positive
    adapter.set_conditioning({"prompt_embeds": embeds * 0 + 7, "image_pad_mask": pad},
                             {"prompt_embeds": embeds * 0 + 9, "image_pad_mask": pad})
    with torch.no_grad():
        adapter.encode_and_prepare(canvas_tensor(rgba((1, 1, 1))), 64, 64, None, torch.device("cpu"))
    assert adapter._branches["cond"][0].eq(7).all() and adapter._branches["uncond"][0].eq(9).all()
    adapter.set_conditioning({"prompt_embeds": embeds * 0 + 5, "image_pad_mask": pad})
    assert adapter._negative is None and adapter._negative_prompt == ""
    prepare(adapter, rgba((164, 58, 207)), prompt="a new request")
    assert adapter.pipe.shown[-1][0] == "a new request" and not adapter._branches["cond"][0].eq(5).all()


def test_injected_conditioning_must_match_the_conditions():
    adapter = make_adapter()
    prepare(adapter, rgba((164, 58, 207)))
    embeds, _, pad = adapter._positive
    adapter.set_conditioning({"prompt_embeds": embeds, "image_pad_mask": pad})
    adapter.set_references([rgba((10, 200, 30))])  # now two conditions, the injected mask covers one
    with pytest.raises(ValueError, match="slot runs"):
        adapter.encode_and_prepare(canvas_tensor(rgba((1, 1, 1))), 64, 64, None, torch.device("cpu"))


# ---- transformer, cache and residency ----


def native_velocity(adapter, x, t, branch, cache, mode):
    """QwenImage21Pipeline's own transformer call (pipeline_qwenimage21.py, denoising loop)."""
    embeds, mask, pad = adapter._branches[branch]
    with adapter.pipe.transformer.cache_context(branch):
        out = adapter.pipe.transformer(
            hidden_states=torch.cat([adapter._condition_latents, x], dim=1),
            timestep=t.expand(x.shape[0]).to(x.dtype) / 1000,
            encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
            img_shapes=adapter._img_shapes, img_mask=pad, attention_kwargs={},
            kv_cache=cache, kv_cache_mode=mode, return_dict=False)[0]
    return out[:, -x.shape[1]:]


@pytest.mark.parametrize("use_kv_cache", [True, False])
def test_velocity_is_the_pipelines_call_bit_for_bit(use_kv_cache):
    adapter = make_adapter()
    adapter.use_kv_cache = use_kv_cache  # causal_condition stays True either way
    adapter.set_references([rgba((10, 200, 30))])
    prepare(adapter, rgba((164, 58, 207)))
    adapter.prepare_timesteps(4, torch.device("cpu"))
    blocks = len(adapter.pipe.transformer.transformer_blocks)
    native = {b: QwenImage21KVCache(blocks) if use_kv_cache else None for b in ("cond", "uncond")}
    adapter._branches["uncond"] = adapter._branch(adapter._encode("", torch.device("cpu")))
    with torch.no_grad():
        for i, (t, sigma) in enumerate(zip(adapter.pipe.scheduler.timesteps, adapter.pipe.scheduler.sigmas)):
            x = torch.randn(1, 16, 64)
            for branch in ("cond", "uncond"):
                mode = ("extract" if i == 0 else "cached") if use_kv_cache else None
                ours = adapter._velocity(x, sigma.item(), branch)
                assert torch.equal(ours, native_velocity(adapter, x, t, branch, native[branch], mode))


def test_prefix_cache_matches_a_full_recompute():
    transformer = tiny_transformer()
    cached, full = make_adapter(transformer), make_adapter(transformer)
    full.use_kv_cache = False
    with torch.no_grad():
        for colour in ((164, 58, 207), (20, 20, 20), (164, 58, 207)):  # A, B, A
            for adapter in (cached, full):
                adapter.set_references([rgba((10, 200, 30))])
                prepare(adapter, rgba(colour))
            assert cached._caches and not full._caches
            for sigma in (1.0, 0.75, 0.2):
                x = torch.randn(1, 16, 64)
                for a, b in zip(cached.predict_x0(x, sigma, 4.0, -0.5), full.predict_x0(x, sigma, 4.0, -0.5)):
                    assert torch.isfinite(a).all()
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def run_request(adapter, canvas, reference, x):
    """One request; returns the cond embeds, condition latents, extracted prefix keys and x0."""
    adapter.set_references([reference])
    prepare(adapter, canvas)
    with torch.no_grad():
        out = adapter.predict_x0(x, 0.9, 4.0, 1.0)
    prefix = [layer.get()[0].clone() for layer in adapter._caches["cond"].layer_caches]
    return adapter._branches["cond"][0].clone(), adapter._condition_latents.clone(), prefix, out


def test_a_resident_adapter_carries_nothing_over():
    adapter = make_adapter()
    x = torch.randn(1, 16, 64)
    canvas = rgba((164, 58, 207))
    embeds_a, cond_a, prefix_a, out_a = run_request(adapter, canvas, rgba((10, 200, 30)), x)
    embeds_b, cond_b, prefix_b, out_b = run_request(adapter, canvas, rgba((200, 120, 40)), x)  # only image2 differs
    assert not torch.equal(embeds_a, embeds_b) and not torch.equal(cond_a, cond_b)
    assert all(not torch.equal(a, b) for a, b in zip(prefix_a, prefix_b))  # the extracted prefix is new
    assert not torch.equal(out_a[0], out_b[0])
    embeds_a2, cond_a2, prefix_a2, out_a2 = run_request(adapter, canvas, rgba((10, 200, 30)), x)
    assert torch.equal(embeds_a, embeds_a2) and torch.equal(cond_a, cond_a2)
    assert all(torch.equal(a, b) for a, b in zip(prefix_a, prefix_a2))
    assert all(torch.equal(a, b) for a, b in zip(out_a, out_a2))


def test_a_resident_adapter_follows_prompt_size_and_references_then_returns_to_a():
    adapter = make_adapter()
    x_a, x_b = torch.randn(1, 16, 64), torch.randn(1, 16, 64)

    def run(prompt, canvas, references, x):
        adapter.set_references(references)
        latent = prepare(adapter, canvas, prompt=prompt).latent
        shown = adapter.pipe.shown[-1]  # the prompt's encoding; the negative follows in predict_x0
        with torch.no_grad():
            out = adapter.predict_x0(x, 0.9, 4.0, 1.0)
        return latent, adapter._img_shapes, shown, out

    a = run("rebuild the earcups", rgba((164, 58, 207)), [rgba((10, 200, 30))], x_a)
    b = run("a wider canvas", rgba((20, 20, 20), size=(128, 32)), [], x_b)  # prompt, size, no reference
    assert b[0].shape == (1, 16, 64) and b[1] == [[(1, 2, 8), (1, 2, 8)]]  # 128x32: 2x8 latents
    assert b[2][0] == "a wider canvas" and len(b[2][1]) == 1  # the old reference is gone
    a2 = run("rebuild the earcups", rgba((164, 58, 207)), [rgba((10, 200, 30))], x_a)
    assert torch.equal(a[0], a2[0]) and a[1] == a2[1]
    assert all(torch.equal(p, q) for p, q in zip(a[3], a2[3]))


def test_a_full_lanpaint_run_counts_wrapper_calls_and_returns_rgba():
    adapter = make_adapter()
    calls = []
    predict = adapter.predict_x0
    adapter.predict_x0 = lambda *args: calls.append(args[1]) or predict(*args)
    keep = Image.new("L", (64, 64), 255)
    keep.paste(0, (16, 16, 48, 48))
    pipe = LanPaintInpaintPipeline(adapter, LanPaintConfig(n_steps=2, blend_overlap=0))
    out = pipe(prompt="edit", image=rgba((164, 58, 207), alpha=0), mask_image=keep,
               num_inference_steps=3, guidance_scale=1.0, seed=4)
    image = out.images[0]
    assert image.mode == "RGBA" and image.size == (64, 64)
    assert len(calls) == 3 + 2 + 2  # one per step, plus 2 inner steps on all but the last
    assert len(set(calls)) == 3


# ---- CLI ----


def load_cli():
    spec = importlib.util.spec_from_file_location("run_lanpaint", Path(__file__).parents[1] / "run_lanpaint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_references_for_models_without_them(monkeypatch, tmp_path):
    cli = load_cli()
    rgba((1, 2, 3)).save(tmp_path / "material.png")
    monkeypatch.setattr(sys, "argv", ["run_lanpaint.py", "--model", "sd3", "--prompt", "x", "--image", "i.png",
                                      "--mask", "m.png", "--reference-image", str(tmp_path / "material.png")])
    monkeypatch.setattr(cli, "create_adapter", lambda *a, **k: pytest.fail("the model must not load"))
    with pytest.raises(ValueError, match="--reference-image"):
        cli.main()


def test_cli_hands_references_to_the_adapter_in_order(monkeypatch, tmp_path):
    cli = load_cli()
    for i, colour in enumerate(((1, 2, 3), (4, 5, 6))):
        rgba(colour, alpha=100).save(tmp_path / f"ref{i}.png")
    adapter = SimpleNamespace(set_references=lambda images: received.extend(images))
    received = []
    monkeypatch.setattr(sys, "argv", ["run_lanpaint.py", "--model", "qwen-image-2.1", "--prompt", "x",
                                      "--image", "i.png", "--mask", "m.png", "--output", str(tmp_path / "o.png"),
                                      "--reference-image", str(tmp_path / "ref0.png"),
                                      "--reference-image", str(tmp_path / "ref1.png")])
    monkeypatch.setattr(cli, "create_adapter", lambda *a, **k: adapter)
    monkeypatch.setattr(cli, "LanPaintInpaintPipeline", lambda *a, **k: (lambda **kw: SimpleNamespace(images=[rgba((0, 0, 0))])))
    cli.main()
    assert [im.getpixel((0, 0)) for im in received] == [(1, 2, 3, 100), (4, 5, 6, 100)]


def test_cli_keeps_an_explicit_zero_guidance(monkeypatch, tmp_path):
    cli = load_cli()
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_lanpaint.py", "--model", "qwen-image-2.1", "--prompt", "x",
                                      "--image", "i.png", "--mask", "m.png", "--output", str(tmp_path / "o.png"),
                                      "--guidance-scale", "0"])
    monkeypatch.setattr(cli, "create_adapter", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli, "LanPaintInpaintPipeline",
                        lambda *a, **k: (lambda **kw: calls.append(kw) or SimpleNamespace(images=[rgba((0, 0, 0))])))
    cli.main()
    assert calls[0]["guidance_scale"] == 0.0


def test_native_single_step_terminal_shift_is_rejected_before_sampling():
    adapter = make_adapter()
    adapter._latent_hw = (64, 64)
    # The checkpoint's terminal stretch divides 0 by 0 for a lone sigma of 1.
    # Keep Diffusers' schedule semantics, but never feed its NaN to LanPaint.
    with pytest.warns(RuntimeWarning, match="invalid value"), pytest.raises(ValueError, match="non-finite"):
        adapter.prepare_timesteps(1, torch.device("cpu"))


@pytest.mark.parametrize("steps", [2, 20])
def test_valid_native_schedules_are_unchanged(steps):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
    adapter = make_adapter()
    adapter._latent_hw = (64, 64)
    native = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
    expected, _ = retrieve_timesteps(native, steps, "cpu", sigmas=np.linspace(1., 1. / steps, steps),
                                    mu=calculate_shift(4096, 256, 8192, .5, .9))
    timesteps, sigmas = adapter.prepare_timesteps(steps, torch.device("cpu"))
    assert torch.equal(timesteps, expected)
    assert torch.equal(sigmas, native.sigmas[:-1])
