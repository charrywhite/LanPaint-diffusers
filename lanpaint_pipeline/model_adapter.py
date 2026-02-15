"""
ModelAdapter — abstract base class for adapting diffusion model pipelines to LanPaint.

Each concrete adapter (FluxKleinAdapter, SD3Adapter, …) implements the model-specific
operations: prompt encoding, VAE encode/decode, transformer predict_x0, timestep setup,
and mask-to-latent conversion.

The generic LanPaintInpaintPipeline delegates all model-specific work to the adapter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image


# ========================= Data containers =========================


@dataclass
class PromptBundle:
    """
    Opaque container for model-specific prompt encodings.

    The pipeline never inspects this — it just passes it through.
    Each adapter stores whatever it needs (embeddings, text_ids, pooled, …).
    """

    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageLatents:
    """
    Encoded image latent + model-specific metadata.

    Attributes:
        latent: The clean image latent in flow notation (Tensor).
        meta:   Adapter-specific metadata needed for decode, mask ops, etc.
                e.g. Flux stores latent_ids, ref_image_ids here.
    """

    latent: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


# ========================= Abstract adapter =========================


class ModelAdapter(ABC):
    """
    Abstract base class for adapting a diffusers pipeline to LanPaint.

    Subclasses implement ~7 methods to handle model-specific details.
    Everything else (denoising loop, LanPaint invocation, mask blend)
    is handled by the generic LanPaintInpaintPipeline.

    Stateful: ``encode_prompt`` and ``encode_and_prepare`` store results
    internally so that ``predict_x0`` can access them without extra args.
    """

    def __init__(self, pipe):
        """
        Parameters
        ----------
        pipe : DiffusionPipeline
            A loaded diffusers pipeline (e.g. Flux2KleinPipeline, SD3Pipeline).
        """
        self.pipe = pipe
        # Populated by encode_prompt / encode_and_prepare
        self._prompt_bundle: Optional[PromptBundle] = None
        self._image_latents: Optional[ImageLatents] = None

    # ---- properties (common across diffusers pipelines) ----

    @property
    def device(self) -> torch.device:
        """Device of the transformer/UNet."""
        model = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        return model.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the transformer/UNet."""
        model = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        return model.dtype

    @property
    def scheduler(self):
        """The diffusers scheduler (shared across all models)."""
        return self.pipe.scheduler

    @property
    def vae_scale_factor(self) -> int:
        """Spatial down-sample factor of the VAE."""
        return self.pipe.vae_scale_factor

    @property
    def image_processor(self):
        """The VaeImageProcessor from the pipeline."""
        return self.pipe.image_processor

    @property
    def requires_ref_at_inference(self) -> bool:
        """
        Whether this model needs the reference image at every denoising step.

        - True: "Edit" style (e.g. Flux Klein). The adapter feeds the ref image
          into the transformer at each step (e.g. concatenated to the latent).
        - False: Standard image-to-image. The ref is only used to initialize
          the noisy latent; during denoising the model only sees the current
          noisy latent (Z-Image, SD3 img2img).
        """
        return False

    # ---- default implementations (flow-matching models) ----

    def noise_scaling(self, sigma, noise, latent_image):
        """
        Flow-matching forward diffusion:  x_t = (1 - t) * x_0 + t * noise.

        Override for non-flow-matching models (e.g. SDXL with epsilon prediction).
        """
        return (1.0 - sigma) * latent_image + sigma * noise

    # ---- abstract methods (must be implemented per model) ----

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        device: torch.device,
    ) -> PromptBundle:
        """
        Encode text prompt(s). Store results internally AND return PromptBundle.
        """
        ...

    @abstractmethod
    def encode_and_prepare(
        self,
        img_tensor: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> ImageLatents:
        """
        VAE-encode the image and prepare all latent metadata.

        Returns ImageLatents with:
          .latent — the clean image latent (flow notation, float32)
          .meta   — adapter-specific metadata (e.g. latent_ids)

        Also stores results internally for predict_x0 / decode_latents to use.
        """
        ...

    @abstractmethod
    def mask_to_latent_space(
        self,
        mask_pixel_keep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Down-sample pixel mask (1, 1, H, W) to latent space.

        Uses stored ``_image_latents.meta`` if needed (e.g. Flux unpacks to get spatial dims).
        Returns mask with shape matching the latent (adapter decides exact shape).
        Convention: 1 = keep, 0 = edit.
        """
        ...

    @abstractmethod
    def prepare_timesteps(
        self,
        num_steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set up the scheduler and return ``(timesteps, flow_ts)``.

        - ``timesteps``: scheduler timestep values (for ``scheduler.step``).
        - ``flow_ts``:  flow-time values in [0, 1] (for LanPaint / predict_x0).
        Both have the same length.
        """
        ...

    @abstractmethod
    def predict_x0(
        self,
        x: torch.Tensor,
        flow_t: float,
        guidance_scale: float,
        cfg_big: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict clean data from noisy latent.

        Uses internally stored prompt and image data.

        Returns
        -------
        x0 : Tensor
            CFG-guided prediction (guidance_scale).
        x0_big : Tensor
            CFG-guided prediction with cfg_big (for LanPaint's score_y).
        """
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latents → PIL image.

        Uses stored ``_image_latents.meta`` for any model-specific decode logic.
        """
        ...
