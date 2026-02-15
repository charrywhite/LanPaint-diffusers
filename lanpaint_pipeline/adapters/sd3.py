"""
SD3Adapter — adapts StableDiffusion3Pipeline to LanPaint's interface.

Handles SD3-specific details:
  - Spatial latent representation (B, C, H, W)
  - Batch concatenation for CFG (not cache-based)
  - scaling_factor / shift_factor for VAE encode/decode
  - encode_prompt returns (embeds, neg_embeds, pooled, neg_pooled)
  - Transformer expects timestep = flow_t * 1000
"""

from typing import Tuple

import torch
from PIL import Image

from diffusers import StableDiffusion3Pipeline

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


class SD3Adapter(ModelAdapter):
    """
    Concrete adapter for StableDiffusion3Pipeline.

    Usage::

        pipe = StableDiffusion3Pipeline.from_pretrained(...)
        adapter = SD3Adapter(pipe)
    """

    def __init__(self, pipe: StableDiffusion3Pipeline):
        super().__init__(pipe)
        # Populated by encode_prompt
        self._prompt_embeds = None
        self._neg_prompt_embeds = None
        self._pooled = None
        self._neg_pooled = None
        # Populated by encode_and_prepare
        self._y_latent = None

    # ---- ModelAdapter implementation ----

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        (
            self._prompt_embeds,
            self._neg_prompt_embeds,
            self._pooled,
            self._neg_pooled,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            device=device,
        )
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "neg_prompt_embeds": self._neg_prompt_embeds,
            "pooled": self._pooled,
            "neg_pooled": self._neg_pooled,
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
        """
        VAE-encode the image.

        SD3 uses: latent = (encode(img).sample() - shift_factor) * scaling_factor
        """
        vae = self.pipe.vae
        with torch.no_grad():
            latent_dist = vae.encode(img_tensor).latent_dist
            y_latent = latent_dist.sample(generator=generator)
            shift = getattr(vae.config, "shift_factor", 0.0)
            scale = vae.config.scaling_factor
            y_latent = (y_latent - shift) * scale

        self._y_latent = y_latent.to(torch.float32)
        self._image_latents = ImageLatents(
            latent=self._y_latent,
            meta={},  # SD3 doesn't need extra metadata
        )
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """
        Pixel mask (1, 1, H, W) → latent mask (1, 1, H_lat, W_lat).

        SD3 latents are spatial, so just a simple interpolation.
        """
        latent_h, latent_w = self._y_latent.shape[-2:]
        mask_latent = torch.nn.functional.interpolate(
            mask_pixel_keep, size=(latent_h, latent_w), mode="nearest",
        ).to(mask_pixel_keep.device, torch.float32)
        return mask_latent

    def prepare_timesteps(
        self, num_steps: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SD3 timestep schedule: simple scheduler.set_timesteps.
        """
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
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
        SD3-specific x0 prediction with dual CFG.

        SD3 transformer uses:
        - Spatial latents (B, C, H, W)
        - Batch concatenation [uncond, cond] for CFG
        - Timestep = flow_t * 1000 (SD3 convention)
        - pooled_projections for conditioning
        """
        model_dtype = self.dtype
        batch_size = x.shape[0]

        # SD3 transformer expects timestep in [0, 1000] range
        timestep_val = flow_t * 1000.0

        # Batch concatenation for CFG
        latent_input = torch.cat([x, x])
        t_input = torch.full((2 * batch_size,), timestep_val, device=x.device, dtype=model_dtype)
        enc_states = torch.cat([self._neg_prompt_embeds, self._prompt_embeds])
        pooled = torch.cat([self._neg_pooled, self._pooled])

        v = self.pipe.transformer(
            hidden_states=latent_input.to(model_dtype),
            timestep=t_input,
            encoder_hidden_states=enc_states,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

        v_uncond, v_cond = v.chunk(2)

        # Dual CFG → x0 via flow matching: x0 = x_t - t * v
        v_cfg = v_uncond.float() + guidance_scale * (v_cond.float() - v_uncond.float())
        v_big = v_uncond.float() + cfg_big * (v_cond.float() - v_uncond.float())

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latents → PIL image.

        SD3: latents → undo scaling/shift → VAE decode → clamp → postprocess.
        """
        model_dtype = self.dtype
        vae = self.pipe.vae

        with torch.no_grad():
            scale = vae.config.scaling_factor
            shift = getattr(vae.config, "shift_factor", 0.0)
            decoded_latents = (latents / scale) + shift
            image = vae.decode(decoded_latents.to(model_dtype), return_dict=False)[0]
            image = image.clamp(-1.0, 1.0)
            pil = self.pipe.image_processor.postprocess(image, output_type="pil")
            return pil[0] if isinstance(pil, list) else pil
