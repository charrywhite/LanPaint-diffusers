"""
ZImageAdapter — adapts ZImageImg2ImgPipeline to LanPaint's interface.

Handles Z-Image–specific details (see diffusers Z-Image img2img pipeline):
  - Spatial latent (B, C, H, W) or 5D (B, C, F, H, W); transformer expects list of (C, F, H, W)
  - Prompt embeds as list of tensors; CFG = prompt_embeds + negative_prompt_embeds
  - Timestep for transformer: (1000 - scheduler_t) / 1000 (0 = noisy, 1 = clean)
  - Flow matching: model output is velocity v; pipeline passes -v to scheduler → x0 = x_t - t*v
  - Timesteps via calculate_shift + retrieve_timesteps(sigmas=..., mu=...)
  - VAE: (latent - shift_factor) * scaling_factor (encode), inverse for decode
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import ZImageImg2ImgPipeline
from diffusers.pipelines.z_image.pipeline_z_image_img2img import (
    calculate_shift,
    retrieve_timesteps,
)

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


def _retrieve_latents(encoder_output, generator: Optional[torch.Generator] = None):
    """Same as pipeline's retrieve_latents for VAE encode."""
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class ZImageAdapter(ModelAdapter):
    """
    Concrete adapter for ZImageImg2ImgPipeline.

    Usage::

        pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", ...)
        adapter = ZImageAdapter(pipe)
    """

    def __init__(self, pipe: ZImageImg2ImgPipeline):
        super().__init__(pipe)
        self._prompt_embeds: Optional[List[torch.Tensor]] = None
        self._neg_prompt_embeds: Optional[List[torch.Tensor]] = None
        self._y_latent: Optional[torch.Tensor] = None
        self._latent_height: int = 0
        self._latent_width: int = 0

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        self._prompt_embeds, self._neg_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            device=device,
            do_classifier_free_guidance=True,
        )
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "neg_prompt_embeds": self._neg_prompt_embeds,
        })
        return self._prompt_bundle

    def encode_and_prepare(
        self,
        img_tensor: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> ImageLatents:
        """VAE-encode image; Z-Image uses (latent - shift_factor) * scaling_factor."""
        vae = self.pipe.vae
        model_dtype = self.dtype

        latent_h = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        self._latent_height, self._latent_width = latent_h, latent_w

        with torch.no_grad():
            img = img_tensor.to(device=device, dtype=model_dtype)
            enc = vae.encode(img)
            y_latent = _retrieve_latents(enc, generator=generator)
            shift = getattr(vae.config, "shift_factor", 0.0)
            scale = getattr(vae.config, "scaling_factor", 1.0)
            y_latent = (y_latent - shift) * scale

        self._y_latent = y_latent.to(torch.float32)
        self._image_latents = ImageLatents(latent=self._y_latent, meta={})
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """Pixel mask (1, 1, H, W) → latent mask (1, 1, H_lat, W_lat)."""
        mask_latent = torch.nn.functional.interpolate(
            mask_pixel_keep,
            size=(self._latent_height, self._latent_width),
            mode="nearest",
        ).to(mask_pixel_keep.device, torch.float32)
        return mask_latent

    def prepare_timesteps(
        self, num_steps: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z-Image: calculate_shift(image_seq_len) + retrieve_timesteps(sigmas=..., mu=mu).

        Returns (scheduler_timesteps, flow_ts) where flow_ts = σ (noise level):
        σ ≈ 1 at the start (noisy) and σ ≈ 0 at the end (clean).
        The model-specific timestep conversion (t_model = 1 − σ) happens in predict_x0.
        """
        image_seq_len = (self._latent_height // 2) * (self._latent_width // 2)
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        if hasattr(self.pipe.scheduler, "sigma_min"):
            self.pipe.scheduler.sigma_min = 0.0

        # Let the scheduler compute its own sigma schedule (sigmas=None).
        # With sigma_min=0 the schedule spans the full [σ_max → 0] range,
        # avoiding a large final jump (explicit linspace stopped at ~0.27).
        # This matches the official Z-Image pipeline: sigmas=None by default.
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler,
            num_steps,
            device,
            mu=mu,
        )
        timesteps = timesteps.to(device)

        # flow_ts = scheduler sigmas = σ (noise level: 1=noisy, 0=clean)
        # This matches the convention used by the Flux Klein adapter and
        # the user's working minimal script: flow_t = t / 1000.
        # Note: scheduler.sigmas has N+1 entries (trailing 0.0 for the final
        # denoising jump).  We take [:-1] (N entries) for the loop; the
        # scheduler's step() uses sigma_next=0 at the last iteration to
        # bring the latent to the fully-clean state.
        flow_ts = self.pipe.scheduler.sigmas.to(device)[:-1]
        timesteps = timesteps[: len(flow_ts)]
        return timesteps, flow_ts

    def predict_x0(
        self,
        x: torch.Tensor,
        flow_t: float,
        guidance_scale: float,
        cfg_big: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Z-Image transformer: list of (C, F, H, W), timestep, list of embeds.

        The input ``x`` can be:
          - 4D (B, C, H, W)        — standard latent
          - 5D (B, C, F, H, W)     — VAE / LanPaint may include a frame dim

        We handle both by:
          1. ``repeat`` along batch with the correct number of dims
          2. Ensure at least 5D (add frame dim=1 if needed) for the transformer
          3. Compute x0 in the original shape

        Z-Image convention vs standard flow matching:
          - Standard: v = noise − x_0 (velocity from clean to noisy)
          - Z-Image model output w = x_0 − noise = −v (velocity from noisy to clean)
          - Therefore: x_0 = x_t + σ * w = x + flow_t * model_output
          - Contrast with Flux/SD3 where x_0 = x − flow_t * v
        """
        model_dtype = self.dtype
        orig_ndim = x.dim()
        batch_size = x.shape[0]

        # Model timestep: t_model = 1 − σ (0 at start/noisy, 1 at end/clean).
        # flow_t = σ (noise level).
        model_t = 1.0 - flow_t
        timestep_tensor = torch.full(
            (batch_size,), model_t, device=x.device, dtype=model_dtype,
        )

        # CFG: repeat along batch dim — match the actual number of dims
        repeat_dims = [2] + [1] * (orig_ndim - 1)  # e.g. [2,1,1,1] for 4D, [2,1,1,1,1] for 5D
        latent_in = x.to(model_dtype).repeat(*repeat_dims)
        timestep_in = timestep_tensor.repeat(2)
        prompt_list = self._prompt_embeds + self._neg_prompt_embeds

        # Transformer expects list of (C, F, H, W).
        # If 4D: add frame dim → (B*2, C, 1, H, W), then unbind → (C, 1, H, W) each
        # If 5D: already (B*2, C, F, H, W), unbind → (C, F, H, W) each
        added_frame_dim = False
        if latent_in.dim() == 4:
            latent_in = latent_in.unsqueeze(2)
            added_frame_dim = True

        latent_list = list(latent_in.unbind(dim=0))  # list of (C, F, H, W)

        model_out_list = self.pipe.transformer(
            latent_list,
            timestep_in,
            prompt_list,
        )[0]

        pos_out = model_out_list[:batch_size]
        neg_out = model_out_list[batch_size:]
        pred_cfg = torch.stack(
            [p.float() + guidance_scale * (p.float() - n.float()) for p, n in zip(pos_out, neg_out)],
            dim=0,
        )  # (B, C, F, H, W)
        pred_big = torch.stack(
            [p.float() + cfg_big * (p.float() - n.float()) for p, n in zip(pos_out, neg_out)],
            dim=0,
        )

        # If we added the frame dim, remove it so shapes match x
        if added_frame_dim:
            pred_cfg = pred_cfg.squeeze(2)
            pred_big = pred_big.squeeze(2)

        # Z-Image: x_0 = x_t + σ * w  (model output w points noisy → clean)
        # Equivalently: noise_pred = -w, then scheduler.step uses -w just like
        # the official pipeline's  `noise_pred = -model_out`.
        x0 = x.float() + flow_t * pred_cfg
        x0_big = x.float() + flow_t * pred_big
        return x0, x0_big

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Z-Image: latents / scaling_factor + shift_factor → VAE decode → postprocess."""
        model_dtype = self.dtype
        vae = self.pipe.vae

        with torch.no_grad():
            scale = getattr(vae.config, "scaling_factor", 1.0)
            shift = getattr(vae.config, "shift_factor", 0.0)
            dec = (latents / scale) + shift
            image = vae.decode(dec.to(model_dtype), return_dict=False)[0]
            image = image.clamp(-1.0, 1.0)
            out = self.pipe.image_processor.postprocess(image, output_type="pil")
            return out[0] if isinstance(out, list) else out
