"""
FluxKleinAdapter — adapts Flux2KleinPipeline to LanPaint's interface.

Handles Flux2-specific details:
  - Packed latent representation (B, L, C)
  - Reference image concatenation to sequence dim
  - Cache-based dual forward pass for CFG
  - BN denorm + unpatchify decode path
  - compute_empirical_mu + retrieve_timesteps for timestep schedule
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


class FluxKleinAdapter(ModelAdapter):
    """
    Concrete adapter for Flux2KleinPipeline.

    Usage::

        pipe = Flux2KleinPipeline.from_pretrained(...)
        adapter = FluxKleinAdapter(pipe)
    """

    # Edit-style model: ref image is concatenated to the latent at every step.
    requires_ref_at_inference = True

    def __init__(self, pipe: Flux2KleinPipeline):
        super().__init__(pipe)
        # Populated by encode_prompt
        self._prompt_embeds = None
        self._text_ids = None
        self._neg_prompt_embeds = None
        self._neg_text_ids = None
        # Populated by encode_and_prepare
        self._y_packed = None
        self._latent_ids = None
        self._ref_image_ids = None

    # ---- ModelAdapter implementation ----

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        self._prompt_embeds, self._text_ids = self.pipe.encode_prompt(
            prompt=prompt, device=device,
        )
        self._neg_prompt_embeds, self._neg_text_ids = self.pipe.encode_prompt(
            prompt=negative_prompt, device=device,
        )
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "text_ids": self._text_ids,
            "neg_prompt_embeds": self._neg_prompt_embeds,
            "neg_text_ids": self._neg_text_ids,
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
        # Encode reference image → packed latent + ref IDs
        image_latents, ref_image_ids = self.pipe.prepare_image_latents(
            images=[img_tensor],
            batch_size=1,
            generator=generator,
            device=device,
            dtype=self.pipe.vae.dtype,
        )
        self._y_packed = image_latents.to(torch.float32)
        self._ref_image_ids = ref_image_ids

        # Prepare noise latent shape + latent IDs (for decode)
        num_ch = self.pipe.transformer.config.in_channels // 4
        _, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_ch,
            height=height,
            width=width,
            dtype=self._prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        self._latent_ids = latent_ids

        self._image_latents = ImageLatents(
            latent=self._y_packed,
            meta={
                "latent_ids": self._latent_ids,
                "ref_image_ids": self._ref_image_ids,
            },
        )
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """
        Pixel mask (1, 1, H, W) → packed latent mask (1, L, 1).

        Flux uses packed sequence representation, so we:
        1. Unpack image latent to get spatial dims (ph, pw)
        2. Interpolate mask to (ph, pw)
        3. Reshape to (1, L, 1)
        """
        y_unpacked = self.pipe._unpack_latents_with_ids(
            self._y_packed, self._ref_image_ids,
        )
        _, _, ph, pw = y_unpacked.shape
        mask_latent = torch.nn.functional.interpolate(
            mask_pixel_keep, size=(ph, pw), mode="nearest",
        ).to(mask_pixel_keep.device, torch.float32).reshape(1, -1, 1)
        return mask_latent

    def prepare_timesteps(
        self, num_steps: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flux2Klein timestep schedule: uses compute_empirical_mu + retrieve_timesteps.
        """
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if getattr(self.pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None

        image_seq_len = self._y_packed.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu,
        )

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
        Flux2-specific x0 prediction with dual CFG.

        Flux transformer uses:
        - Packed sequence: [noisy_latent ; ref_image]
        - Cache-context based conditional / unconditional passes
        - Flow matching: x0 = x_t - t * v
        """
        seq_len = x.shape[1]
        model_dtype = self.dtype

        # Concatenate reference image to sequence
        latent_model_input = torch.cat(
            [x, self._y_packed.to(x.device, x.dtype)], dim=1,
        )
        img_ids = torch.cat([self._latent_ids, self._ref_image_ids], dim=1)
        t_tensor = torch.full(
            (x.shape[0],), flow_t, device=x.device, dtype=model_dtype,
        )

        # Conditional pass
        with self.pipe.transformer.cache_context("cond"):
            v_cond = self.pipe.transformer(
                hidden_states=latent_model_input.to(model_dtype),
                timestep=t_tensor,
                guidance=None,
                encoder_hidden_states=self._prompt_embeds,
                txt_ids=self._text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0][:, :seq_len]

        # Unconditional pass
        with self.pipe.transformer.cache_context("uncond"):
            v_uncond = self.pipe.transformer(
                hidden_states=latent_model_input.to(model_dtype),
                timestep=t_tensor,
                guidance=None,
                encoder_hidden_states=self._neg_prompt_embeds,
                txt_ids=self._neg_text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0][:, :seq_len]

        # Dual CFG → x0 via flow matching
        v_cfg = v_uncond.float() + guidance_scale * (v_cond.float() - v_uncond.float())
        v_big = v_uncond.float() + cfg_big * (v_cond.float() - v_uncond.float())

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode packed latents → PIL image.

        Flux2-specific: unpack → BN denorm → unpatchify → VAE decode.
        """
        model_dtype = self.dtype

        with torch.no_grad():
            unpacked = self.pipe._unpack_latents_with_ids(latents, self._latent_ids)

            # BN denorm
            bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
                unpacked.device, unpacked.dtype,
            )
            bn_std = torch.sqrt(
                self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
            ).to(unpacked.device, unpacked.dtype)
            unpacked = unpacked * bn_std + bn_mean

            # Unpatchify + decode
            spatial = self.pipe._unpatchify_latents(unpacked)
            img = self.pipe.vae.decode(spatial.to(model_dtype), return_dict=False)[0]
            pil = self.pipe.image_processor.postprocess(img, output_type="pil")
            return pil[0] if isinstance(pil, list) else pil
