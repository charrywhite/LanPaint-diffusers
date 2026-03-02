"""
QwenAdapter — adapts QwenImageEditPlusPipeline to LanPaint's interface.

Handles Qwen-specific details:
  - Packed latent representation (B, L, C*4=64) via 2x2 patching
  - Reference image concatenation to sequence dim at every denoising step
  - Vision-language prompt encoding (needs condition image)
  - Cache-based dual forward pass for CFG + norm rescaling
  - Mean/std VAE normalization (latents_mean, latents_std vectors)
  - calculate_shift + retrieve_timesteps for timestep schedule

The prompt-encoding challenge:
  ModelAdapter.encode_prompt() is called BEFORE encode_and_prepare() by the
  LanPaint pipeline, but Qwen's VL encoder needs a condition image.
  Solution: encode_prompt() stores raw strings; actual encoding happens
  inside encode_and_prepare() once the image is available.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift,
    retrieve_timesteps,
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


def _retrieve_latents(encoder_output, generator: Optional[torch.Generator] = None):
    """Extract latents from VAE encoder output (argmax mode for determinism)."""
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class QwenAdapter(ModelAdapter):
    """
    Concrete adapter for QwenImageEditPlusPipeline.

    Usage::

        pipe = QwenImageEditPlusPipeline.from_pretrained(...)
        adapter = QwenAdapter(pipe)
    """

    # Edit-style model: ref image is concatenated to the latent at every step.
    requires_ref_at_inference = True

    def __init__(self, pipe: QwenImageEditPlusPipeline):
        super().__init__(pipe)
        # Populated by encode_prompt (deferred — stores raw strings)
        self._prompt_str: Optional[str] = None
        self._neg_prompt_str: Optional[str] = None
        # Populated by encode_and_prepare (actual encoding happens here)
        self._prompt_embeds = None
        self._prompt_embeds_mask = None
        self._neg_prompt_embeds = None
        self._neg_prompt_embeds_mask = None
        self._image_latents_packed = None  # packed ref image latents (B, L, C*4)
        self._img_shapes = None
        self._latent_height: int = 0
        self._latent_width: int = 0
        self._pixel_height: int = 0
        self._pixel_width: int = 0
        self._conditioning_preloaded: bool = False

    # ---- ModelAdapter implementation ----

    def set_conditioning(self, positive_conditioning, negative_conditioning=None):
        """
        Inject pre-encoded Qwen conditioning dict.

        positive_conditioning: dict with prompt_embeds, prompt_embeds_mask, image.
        negative_conditioning: optional dict with prompt_embeds, prompt_embeds_mask.
        """
        self._prompt_embeds = positive_conditioning["prompt_embeds"]
        self._prompt_embeds_mask = positive_conditioning["prompt_embeds_mask"]

        if negative_conditioning is not None:
            self._neg_prompt_embeds = negative_conditioning["prompt_embeds"]
            self._neg_prompt_embeds_mask = negative_conditioning["prompt_embeds_mask"]
        else:
            self._neg_prompt_embeds = torch.zeros_like(self._prompt_embeds)
            self._neg_prompt_embeds_mask = torch.zeros_like(self._prompt_embeds_mask)

        self._conditioning_preloaded = True
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "prompt_embeds_mask": self._prompt_embeds_mask,
        })

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        """
        Store raw prompt strings for deferred encoding.

        Qwen's VL text encoder requires a condition image, which isn't available
        until encode_and_prepare(). We store the strings here and do the actual
        encoding in encode_and_prepare().
        """
        self._prompt_str = prompt
        self._neg_prompt_str = negative_prompt
        self._prompt_bundle = PromptBundle(data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
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
        VAE-encode the image, encode prompts with VL encoder, prepare packed latents.

        Qwen edit pipeline flow:
        1. Reconstruct PIL from img_tensor for VL encoder condition image
        2. Encode prompt + negative_prompt with VL encoder + condition image
        3. VAE-encode img_tensor → normalize with mean/std → pack to (B, L, 64)
        4. Store ref image latents for predict_x0 concatenation
        """
        pipe = self.pipe
        model_dtype = self.dtype

        self._pixel_height = height
        self._pixel_width = width

        # Latent spatial dims (before packing: H_lat x W_lat)
        vae_sf = pipe.vae_scale_factor
        self._latent_height = 2 * (int(height) // (vae_sf * 2))
        self._latent_width = 2 * (int(width) // (vae_sf * 2))

        # --- 1 & 2. Encode prompts (skip if conditioning was pre-loaded) ---
        if not self._conditioning_preloaded:
            # Prepare condition image for VL encoder
            # img_tensor is (B, C, H, W) in [-1, 1]; reconstruct PIL for VL encoder
            img_np = img_tensor[0].detach().float().cpu()
            img_np = ((img_np / 2.0) + 0.5).clamp(0.0, 1.0)
            img_np = img_np.permute(1, 2, 0).numpy()
            condition_pil = Image.fromarray((img_np * 255).astype(np.uint8))

            # Resize for VL encoder (384x384 target area)
            img_w, img_h = condition_pil.size
            cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
            condition_image = pipe.image_processor.resize(condition_pil, cond_h, cond_w)

            # Encode prompts with VL encoder
            self._prompt_embeds, self._prompt_embeds_mask = pipe.encode_prompt(
                prompt=self._prompt_str,
                image=[condition_image],
                device=device,
            )

            # Always encode the negative prompt (even empty string) so CFG is active.
            # LanPaint needs both cond and uncond predictions for proper inpainting.
            neg_str = self._neg_prompt_str if self._neg_prompt_str is not None else ""
            self._neg_prompt_embeds, self._neg_prompt_embeds_mask = pipe.encode_prompt(
                prompt=neg_str,
                image=[condition_image],
                device=device,
            )

        # --- 3. VAE-encode at target resolution ---
        # img_tensor is already preprocessed at (height, width); add time dim for Qwen VAE
        vae_input = img_tensor.unsqueeze(2).to(device=device, dtype=model_dtype)  # (B, C, 1, H, W)

        with torch.no_grad():
            image_latents = _retrieve_latents(pipe.vae.encode(vae_input), generator=generator)

        # Normalize with per-channel mean/std
        latent_channels = pipe.vae.config.z_dim
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        # Pack: (B, C, 1, H_lat, W_lat) → (B, L, C*4=64)
        lat_h, lat_w = image_latents.shape[3], image_latents.shape[4]
        image_latents_packed = pipe._pack_latents(
            image_latents, 1, latent_channels, lat_h, lat_w,
        )
        self._image_latents_packed = image_latents_packed.to(torch.float32)

        # --- 4. Build img_shapes for transformer ---
        # Format: [[noise_shape, ref_image_shape]] per batch element
        noise_patch_h = self._latent_height // 2
        noise_patch_w = self._latent_width // 2
        ref_patch_h = lat_h // 2
        ref_patch_w = lat_w // 2
        self._img_shapes = [
            [
                (1, noise_patch_h, noise_patch_w),
                (1, ref_patch_h, ref_patch_w),
            ]
        ]

        self._image_latents = ImageLatents(
            latent=self._image_latents_packed,
            meta={
                "latent_height": self._latent_height,
                "latent_width": self._latent_width,
                "pixel_height": self._pixel_height,
                "pixel_width": self._pixel_width,
            },
        )
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """
        Pixel mask (1, 1, H, W) → packed latent mask (1, L, 1).

        Qwen uses packed sequence representation (2x2 patches), so we:
        1. Interpolate mask to spatial latent dims (latent_height, latent_width)
        2. Pool 2x2 patches and reshape to (1, L, 1)
        """
        mask_latent = torch.nn.functional.interpolate(
            mask_pixel_keep,
            size=(self._latent_height, self._latent_width),
            mode="nearest",
        ).to(mask_pixel_keep.device, torch.float32)

        # Pack 2x2 patches: (1, 1, H, W) → (1, (H/2)*(W/2), 1)
        # If the majority of pixels in a 2x2 patch are "keep", the patch is "keep"
        _, _, h, w = mask_latent.shape
        mask_latent = mask_latent.view(1, 1, h // 2, 2, w // 2, 2)
        mask_latent = mask_latent.mean(dim=(1, 3, 5))  # (1, h//2, w//2)
        mask_latent = (mask_latent > 0.5).float()
        mask_latent = mask_latent.reshape(1, -1, 1)

        return mask_latent

    def prepare_timesteps(
        self, num_steps: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Qwen timestep schedule: calculate_shift(noise_seq_len) + retrieve_timesteps(sigmas=linspace, mu=mu).

        Matches the Qwen pipeline: sigmas = linspace(1.0, 1/N, N), shifted by mu.
        """
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)

        noise_seq_len = (self._latent_height // 2) * (self._latent_width // 2)
        mu = calculate_shift(
            noise_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
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
        Qwen-specific x0 prediction with dual CFG + norm rescaling.

        Qwen transformer uses:
        - Packed sequence: [noisy_latent ; ref_image_latents] along dim=1
        - flow_t passed directly as timestep (pipeline divides by 1000 internally)
        - Cache-context based conditional / unconditional passes
        - CFG with norm rescaling: combined = uncond + scale*(cond - uncond),
          then rescale: v = combined * (||cond|| / ||combined||)
        - Flow matching: x0 = x_t - t * v
        """
        seq_len = x.shape[1]
        model_dtype = self.dtype

        # Concatenate reference image latents to sequence
        latent_model_input = torch.cat(
            [x, self._image_latents_packed.to(x.device, x.dtype)], dim=1,
        )

        # Qwen transformer expects timestep = sigma (the pipeline passes t/1000,
        # where t is the scheduler timestep = sigma * 1000)
        timestep = torch.full(
            (x.shape[0],), flow_t, device=x.device, dtype=model_dtype,
        )

        # Guidance embedding (for guidance-distilled variants)
        guidance = None
        if getattr(self.pipe.transformer.config, "guidance_embeds", False):
            guidance = torch.full(
                [x.shape[0]], guidance_scale, device=x.device, dtype=torch.float32,
            )

        # Conditional pass
        with self.pipe.transformer.cache_context("cond"):
            v_cond = self.pipe.transformer(
                hidden_states=latent_model_input.to(model_dtype),
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=self._prompt_embeds,
                encoder_hidden_states_mask=self._prompt_embeds_mask,
                img_shapes=self._img_shapes,
                return_dict=False,
            )[0][:, :seq_len]

        # Unconditional pass — always run for LanPaint (needs both predictions)
        with self.pipe.transformer.cache_context("uncond"):
            v_uncond = self.pipe.transformer(
                hidden_states=latent_model_input.to(model_dtype),
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=self._neg_prompt_embeds,
                encoder_hidden_states_mask=self._neg_prompt_embeds_mask,
                img_shapes=self._img_shapes,
                return_dict=False,
            )[0][:, :seq_len]

        # CFG with norm rescaling (matches Qwen pipeline exactly)
        v_cond_f = v_cond.float()
        v_uncond_f = v_uncond.float()

        v_cfg = v_uncond_f + guidance_scale * (v_cond_f - v_uncond_f)
        v_big_raw = v_uncond_f + cfg_big * (v_cond_f - v_uncond_f)

        # Norm rescaling: preserve the conditional prediction's norm
        cond_norm = torch.norm(v_cond_f, dim=-1, keepdim=True)
        cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
        big_norm = torch.norm(v_big_raw, dim=-1, keepdim=True).clamp(min=1e-8)

        v_cfg = v_cfg * (cond_norm / cfg_norm)
        v_big = v_big_raw * (cond_norm / big_norm)

        # Flow matching: x0 = x_t - t * v
        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode packed latents → PIL image.

        Qwen-specific: unpack → denormalize (mean/std) → VAE decode → postprocess.
        """
        pipe = self.pipe
        model_dtype = self.dtype

        with torch.no_grad():
            # Unpack: (B, L, C*4) → (B, C, 1, H, W)
            unpacked = pipe._unpack_latents(
                latents,
                self._pixel_height,
                self._pixel_width,
                pipe.vae_scale_factor,
            )

            # Denormalize: latent = latent / (1/std) + mean
            # Matches pipeline decode: latents / latents_std_inv + latents_mean
            latent_channels = pipe.vae.config.z_dim
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, latent_channels, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            latents_std_inv = (
                1.0 / torch.tensor(pipe.vae.config.latents_std)
                .view(1, latent_channels, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            unpacked = unpacked / latents_std_inv + latents_mean

            # VAE decode — Qwen VAE outputs (B, C, T, H, W), take frame 0
            img = pipe.vae.decode(unpacked.to(model_dtype), return_dict=False)[0][:, :, 0]
            pil = pipe.image_processor.postprocess(img, output_type="pil")
            return pil[0] if isinstance(pil, list) else pil
