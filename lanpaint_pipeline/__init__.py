"""
LanPaint multi-model inpainting pipeline.

Architecture:
  Layer 1: ModelAdapter (ABC)    — per-model, handles transformer/VAE/prompt specifics
  Layer 2: LanPaintModelWrapper  — generic bridge adapting any ModelAdapter to LanPaint's interface
  Layer 3: LanPaintInpaintPipeline — generic orchestrator (denoising loop, mask, blend)

Usage:
    from lanpaint_pipeline import LanPaintInpaintPipeline, LanPaintConfig
    from lanpaint_pipeline.adapters.flux_klein import FluxKleinAdapter

    pipe = Flux2KleinPipeline.from_pretrained(...)
    adapter = FluxKleinAdapter(pipe)
    lp_pipe = LanPaintInpaintPipeline(adapter, config=LanPaintConfig(...))
    result = lp_pipe(prompt=..., image=..., mask_image=...)
"""

from lanpaint_pipeline.model_adapter import (
    ImageLatents,
    ModelAdapter,
    PromptBundle,
)
from lanpaint_pipeline.pipeline import (
    LanPaintConfig,
    LanPaintInpaintPipeline,
    LanPaintModelWrapper,
    LanPaintOutput,
)
from lanpaint_pipeline.registry import (
    create_adapter,
    get_model_spec,
    list_models,
    register_model,
    ModelSpec,
)
from lanpaint_pipeline.utils import (
    blend_with_smooth_mask,
    flow_to_abt,
    flow_to_ve_sigma,
    gaussian_kernel_2d,
    load_image_preserve_alpha,
    make_current_times,
)

__all__ = [
    "ModelAdapter",
    "PromptBundle",
    "ImageLatents",
    "LanPaintConfig",
    "LanPaintInpaintPipeline",
    "LanPaintModelWrapper",
    "LanPaintOutput",
    "flow_to_abt",
    "flow_to_ve_sigma",
    "make_current_times",
    "gaussian_kernel_2d",
    "blend_with_smooth_mask",
    "load_image_preserve_alpha",
    "create_adapter",
    "get_model_spec",
    "list_models",
    "register_model",
    "ModelSpec",
]
