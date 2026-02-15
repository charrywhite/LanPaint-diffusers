"""
Model registry — maps model names to (pipeline_class, adapter_class, defaults).

Adding a new model = adding one entry to MODEL_REGISTRY. No new scripts needed.

Usage:
    from lanpaint_pipeline.registry import create_adapter, list_models

    adapter = create_adapter("flux-klein", device="cuda")
    # or
    adapter = create_adapter("sd3", device="cuda", model_id="my-custom-sd3-checkpoint")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

import torch

from lanpaint_pipeline.model_adapter import ModelAdapter


@dataclass
class ModelSpec:
    """Specification for a registered model."""

    # Human-readable name
    name: str
    # Dot-path to diffusers pipeline class (lazy import to avoid loading all of diffusers)
    pipeline_cls_path: str
    # Dot-path to adapter class
    adapter_cls_path: str
    # Default HuggingFace model ID
    default_model_id: str
    # Default torch dtype
    default_dtype: torch.dtype = torch.bfloat16
    # Default inference parameters (guidance_scale, num_inference_steps, etc.)
    default_params: Dict[str, Any] = field(default_factory=dict)
    # True = edit-style (ref image fed to model every step, e.g. Flux Klein);
    # False = standard img2img (ref only used to init latent, e.g. Z-Image, SD3).
    requires_ref_at_inference: bool = False


def _import_class(dotpath: str):
    """Import a class from a dot-separated path like 'diffusers.Flux2KleinPipeline'."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


# ========================= Registry =========================

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(key: str, spec: ModelSpec):
    """Register a model specification."""
    MODEL_REGISTRY[key] = spec


def list_models() -> list[str]:
    """Return list of registered model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model_spec(key: str) -> ModelSpec:
    """Look up a model spec by key. Raises KeyError if not found."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{key}'. Available: {available}")
    return MODEL_REGISTRY[key]


def create_adapter(
    model_key: str,
    *,
    device: str = "cuda",
    model_id: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **from_pretrained_kwargs: Any,
) -> ModelAdapter:
    """
    One-liner to create a ready-to-use adapter.

    Parameters
    ----------
    model_key : str
        Registered model key (e.g. "flux-klein", "sd3", "z-image").
    device : str
        Target device.
    model_id : str, optional
        Override the default HuggingFace model ID or path to local checkpoint.
    torch_dtype : torch.dtype, optional
        Override the default dtype.
    **from_pretrained_kwargs
        Passed through to ``PipelineClass.from_pretrained()`` (e.g. ``local_files_only=True``,
        ``low_cpu_mem_usage=False`` for Z-Image when using a local path).

    Returns
    -------
    ModelAdapter
        Fully initialized adapter with the diffusers pipeline loaded and on device.
    """
    spec = get_model_spec(model_key)
    PipelineClass = _import_class(spec.pipeline_cls_path)
    AdapterClass = _import_class(spec.adapter_cls_path)

    effective_model_id = model_id or spec.default_model_id
    effective_dtype = torch_dtype or spec.default_dtype

    pipe = PipelineClass.from_pretrained(
        effective_model_id,
        torch_dtype=effective_dtype,
        **from_pretrained_kwargs,
    )
    pipe.to(device)

    return AdapterClass(pipe)


# ========================= Built-in registrations =========================

register_model("flux-klein", ModelSpec(
    name="Flux2 Klein 9B",
    pipeline_cls_path="diffusers.Flux2KleinPipeline",
    adapter_cls_path="lanpaint_pipeline.adapters.flux_klein.FluxKleinAdapter",
    default_model_id="black-forest-labs/FLUX.2-klein-base-9B",
    default_dtype=torch.bfloat16,
    default_params={"guidance_scale": 5.0, "num_inference_steps": 20},
    requires_ref_at_inference=True,  # edit-style: ref image in transformer every step
))

register_model("sd3", ModelSpec(
    name="Stable Diffusion 3 Medium",
    pipeline_cls_path="diffusers.StableDiffusion3Pipeline",
    adapter_cls_path="lanpaint_pipeline.adapters.sd3.SD3Adapter",
    default_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    default_dtype=torch.float16,
    default_params={"guidance_scale": 7.0, "num_inference_steps": 50},
))

register_model("z-image", ModelSpec(
    name="Z-Image Turbo",
    pipeline_cls_path="diffusers.ZImageImg2ImgPipeline",
    adapter_cls_path="lanpaint_pipeline.adapters.z_image.ZImageAdapter",
    default_model_id="Tongyi-MAI/Z-Image-Turbo",
    default_dtype=torch.bfloat16,
    default_params={"guidance_scale": 5.0, "num_inference_steps": 20},
))
