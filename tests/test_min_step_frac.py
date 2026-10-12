"""The tail ramp is checked against hand-computed cases and the real LanPaint core.

The integration fixture replaces only the model/VAE: it runs the real pipeline,
Euler scheduler and Langevin updates, without downloading model weights.
"""
import numpy as np
import pytest
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor

import lanpaint_pipeline.pipeline as pipeline_module
from lanpaint_pipeline.model_adapter import ImageLatents
from lanpaint_pipeline.pipeline import LanPaintConfig, LanPaintInpaintPipeline
from lanpaint_pipeline.utils import min_step_frac_effective_steps


@pytest.mark.parametrize("count,noise,threshold,expected", [
    (5, 0.9, 0.0, 5), (5, 0.0, 0.0, 5),  # disabled, including the tail
    (5, 0.75, 0.5, 5), (5, 0.5, 0.5, 5),  # unchanged at/above threshold
    (5, 0.25, 0.5, 2), (5, 0.35, 0.5, 4),  # round-to-even ties, as upstream
    (5, 0.01, 1.0, 0), (0, 0.5, 1.0, 0),
])
def test_inner_step_ramp(count, noise, threshold, expected):
    assert min_step_frac_effective_steps(count, noise, threshold) == expected


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_fraction_is_rejected(value):
    with pytest.raises(ValueError, match="min_step_frac"):
        LanPaintConfig(min_step_frac=value)
    with pytest.raises(ValueError, match="min_step_frac"):
        min_step_frac_effective_steps(5, 0.5, value)


def test_config_preserves_existing_positional_arguments():
    config = LanPaintConfig(2, 15.0, 8.0, 1.0, 0.2, 1, 1.0, 9)
    assert config.blend_overlap == 9
    assert config.min_step_frac == 0.0


class ConstantVelocityAdapter:
    supports_alpha = False
    device = torch.device("cpu")
    dtype = torch.float32
    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    def __init__(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True, shift_terminal=0.02,
        )
        self.model_calls = 0
        self.clean = torch.linspace(-0.5, 0.5, 64).reshape(1, 16, 4)

    def encode_prompt(self, *args):
        return None

    def encode_and_prepare(self, *args):
        return ImageLatents(self.clean.clone())

    def mask_to_latent_space(self, mask):
        return torch.nn.functional.interpolate(mask, size=(4, 4), mode="nearest").flatten(2).transpose(1, 2)

    def prepare_timesteps(self, count, device):
        # Qwen 2.1's 1024-square schedule, supplied independently of the new ramp.
        mu = 0.5 + (4096 - 256) * (0.9 - 0.5) / (8192 - 256)
        self.scheduler.set_timesteps(sigmas=np.linspace(1, 1 / count, count), mu=mu, device=device)
        return self.scheduler.timesteps, self.scheduler.sigmas[:-1]

    def noise_scaling(self, sigma, noise, latent):
        return (1 - sigma) * latent + sigma * noise

    def predict_x0(self, x, flow_t, guidance_scale, cfg_big):
        self.model_calls += 1
        out = x - flow_t * 0.125
        return out, out

    def decode_latents(self, latents):
        self.decoded = latents.clone()
        return Image.new("RGB", (32, 32))


def run_real_core(monkeypatch, config):
    base = pipeline_module.LanPaint
    recorded = []

    class ObservedLanPaint(base):
        def __call__(self, *args, **kwargs):
            self.iterations = 0
            self.step_sizes = []
            count = kwargs["n_steps"]
            result = super().__call__(*args, **kwargs)
            recorded.append((self.n_steps if count is None else count, self.iterations, self.min_step_frac, self.step_sizes))
            return result

        def langevin_dynamics(self, *args, **kwargs):
            self.iterations += 1
            self.step_sizes.append(float(args[3].flatten()[0]))
            return super().langevin_dynamics(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(pipeline_module, "LanPaint", ObservedLanPaint)
        adapter = ConstantVelocityAdapter()
        mask = Image.new("L", (32, 32), 255)
        mask.paste(0, (16, 0, 32, 32))
        pipe = LanPaintInpaintPipeline(adapter, config)
        with torch.random.fork_rng():
            pipe(prompt="edit", image=Image.new("RGB", (32, 32)), mask_image=mask,
                 num_inference_steps=20, seed=17, height=32, width=32)
    return adapter, recorded


@pytest.mark.parametrize("threshold,expected", [
    (0.0, [5] * 19 + [0]),
    (1.0, [5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0]),
])
def test_real_core_ramp_and_last_step_preservation(monkeypatch, threshold, expected):
    config = LanPaintConfig(n_steps=5, min_step_frac=threshold, blend_overlap=0)
    adapter, recorded = run_real_core(monkeypatch, config)
    assert [r[0] for r in recorded] == expected
    assert [r[1] for r in recorded] == expected  # actual Langevin updates, not just requested counts
    assert all(r[2] == threshold for r in recorded)
    assert adapter.model_calls == 20 + sum(expected)
    assert torch.isfinite(adapter.decoded).all()
    keep = torch.tensor([True, True, False, False] * 4)
    torch.testing.assert_close(adapter.decoded[:, keep], adapter.clean[:, keep], atol=1e-4, rtol=0)
    if threshold == 1.0:
        sizes = [size for r in recorded for size in r[3]]
        assert sizes == pytest.approx([config.step_size] * sum(expected))


def test_early_stop_overrides_ramp(monkeypatch):
    adapter, recorded = run_real_core(monkeypatch, LanPaintConfig(n_steps=5, min_step_frac=1.0, early_stop=20, blend_overlap=0))
    assert [r[1] for r in recorded] == [0] * 20
    assert adapter.model_calls == 20


def test_default_path_runs_with_a_lanpaint_that_predates_the_ramp(monkeypatch):
    class Built(Exception):
        pass

    class OlderLanPaint:
        """Constructor of a LanPaint release without MinStepFrac."""
        def __init__(self, Model, NSteps, Friction, Lambda, Beta, StepSize, IS_FLUX=False, IS_FLOW=False):
            raise Built

    monkeypatch.setattr(pipeline_module, "LanPaint", OlderLanPaint)
    mask = Image.new("L", (32, 32), 255)
    for threshold, error, match in ((0.0, Built, None), (1.0, TypeError, "MinStepFrac")):
        pipe = LanPaintInpaintPipeline(ConstantVelocityAdapter(), LanPaintConfig(min_step_frac=threshold))
        with pytest.raises(error, match=match):
            pipe(prompt="edit", image=Image.new("RGB", (32, 32)), mask_image=mask,
                 num_inference_steps=20, seed=17, height=32, width=32)


def test_cli_rejects_bad_fraction_before_loading_model(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_lanpaint", Path(__file__).parents[1] / "run_lanpaint.py")
    run_lanpaint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_lanpaint)
    monkeypatch.setattr("sys.argv", ["run_lanpaint.py", "--prompt", "edit", "--image", "unused.png",
                                    "--mask", "unused-mask.png", "--lp-min-step-frac", "nan"])
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid MinStepFrac must fail before loading the model")
    monkeypatch.setattr(run_lanpaint, "create_adapter", unexpected_load)
    with pytest.raises(ValueError, match="min_step_frac"):
        run_lanpaint.main()
