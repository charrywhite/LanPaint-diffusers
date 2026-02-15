"""
LanPaintInpaintPipeline — model-agnostic orchestrator for LanPaint inpainting.

Coordinates:
  1. Image/mask preprocessing
  2. Prompt & image encoding (via adapter)
  3. LanPaint Langevin dynamics
  4. Euler denoising loop
  5. VAE decode (via adapter)
  6. Smooth mask blend post-processing
"""

from dataclasses import dataclass
import os
import re
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from LanPaint.lanpaint import LanPaint

from lanpaint_pipeline.model_adapter import ModelAdapter
from lanpaint_pipeline.utils import (
    blend_with_smooth_mask,
    load_image_preserve_alpha,
    make_current_times,
)


@dataclass
class LanPaintConfig:
    """LanPaint hyper-parameters. Maps directly to LanPaint constructor args."""

    n_steps: int = 2
    friction: float = 15.0
    chara_lambda: float = 8.0
    beta: float = 1.0
    step_size: float = 0.2
    early_stop: int = 1
    cfg_big: float = 1.0
    blend_overlap: int = 9


class LanPaintModelWrapper:
    """
    Generic bridge: adapts any ModelAdapter -> LanPaint's expected model interface.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        guidance_scale: float,
        cfg_big: float = 1.0,
    ):
        self.adapter = adapter
        self.guidance_scale = guidance_scale
        self.cfg_big = cfg_big

        # LanPaint expects model.inner_model.model_sampling.noise_scaling
        self.inner_model = self
        self.model_sampling = self

    def noise_scaling(self, sigma, noise, latent_image):
        return self.adapter.noise_scaling(sigma, noise, latent_image)

    def __call__(self, x, t, model_options=None, seed=None):
        flow_t = float(t.flatten()[0])
        return self.adapter.predict_x0(x, flow_t, self.guidance_scale, self.cfg_big)


@dataclass
class LanPaintOutput:
    images: list


class LanPaintInpaintPipeline:
    """Model-agnostic LanPaint inpainting pipeline."""

    def __init__(self, adapter: ModelAdapter, config: Optional[LanPaintConfig] = None):
        self.adapter = adapter
        self.config = config or LanPaintConfig()

    @classmethod
    def from_adapter(cls, adapter: ModelAdapter, config: Optional[LanPaintConfig] = None):
        return cls(adapter, config)

    def __call__(
        self,
        *,
        prompt: str,
        image: Union[str, Image.Image],
        mask_image: Optional[Union[str, Image.Image]] = None,
        outpaint_padding: Optional[str] = None,
        save_preprocess_dir: Optional[str] = None,
        negative_prompt: str = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        seed: int = 0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> LanPaintOutput:
        device = self.adapter.device
        config = self.config

        orig_img, mask_keep_pixel, img_tensor, image_height, image_width = self._preprocess(
            image=image,
            mask_image=mask_image,
            outpaint_padding=outpaint_padding,
            height=height,
            width=width,
            device=device,
        )
        if save_preprocess_dir:
            self._save_preprocess_debug(save_preprocess_dir, orig_img, img_tensor, mask_keep_pixel)

        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            _ = self.adapter.encode_prompt(prompt, negative_prompt, device)

        with torch.no_grad():
            image_latents = self.adapter.encode_and_prepare(
                img_tensor, image_height, image_width, generator, device
            )
        y_latent = image_latents.latent

        with torch.no_grad():
            mask_keep_latent = self.adapter.mask_to_latent_space(mask_keep_pixel)

        model_wrapper = LanPaintModelWrapper(
            adapter=self.adapter,
            guidance_scale=guidance_scale,
            cfg_big=config.cfg_big,
        )

        lanpaint = LanPaint(
            Model=model_wrapper,
            NSteps=config.n_steps,
            Friction=config.friction,
            Lambda=config.chara_lambda,
            Beta=config.beta,
            StepSize=config.step_size,
            IS_FLUX=False,
            IS_FLOW=True,
        )

        timesteps, flow_ts = self.adapter.prepare_timesteps(num_inference_steps, device)
        num_steps = len(timesteps)

        noise = torch.randn_like(y_latent)
        latents = model_wrapper.noise_scaling(
            flow_ts[0:1].reshape(1, 1, 1).float(), noise, y_latent
        )

        if hasattr(self.adapter.scheduler, "set_begin_index"):
            self.adapter.scheduler.set_begin_index(0)

        with torch.no_grad():
            for i, (t, flow_t) in enumerate(
                tqdm(zip(timesteps, flow_ts), desc="LanPaint", total=num_steps)
            ):
                flow_t_val = flow_t.item()
                current_times = make_current_times(flow_t_val, device)
                n_steps_override = 0 if (num_steps - i <= config.early_stop) else None

                x0 = lanpaint(
                    x=latents,
                    latent_image=y_latent,
                    noise=noise,
                    sigma=torch.tensor([flow_t_val], device=device, dtype=torch.float32),
                    latent_mask=mask_keep_latent,
                    current_times=current_times,
                    model_options={},
                    seed=seed,
                    n_steps=n_steps_override,
                )

                noise_pred = (latents - x0) / max(flow_t_val, 1e-6)
                latents = self.adapter.scheduler.step(
                    noise_pred.to(torch.float32),
                    t,
                    latents.to(torch.float32),
                    return_dict=False,
                )[0].to(torch.float32)

        decoded_image = self.adapter.decode_latents(latents)
        if save_preprocess_dir and output_type == "pil":
            os.makedirs(save_preprocess_dir, exist_ok=True)
            decoded_image.save(os.path.join(save_preprocess_dir, "pre_blend_decoded_image.png"))

        if output_type == "pil" and config.blend_overlap > 0:
            decoded_image = blend_with_smooth_mask(
                orig_img,
                decoded_image,
                mask_keep_pixel,
                overlap=config.blend_overlap,
                device=device,
            )

        return LanPaintOutput(images=[decoded_image])

    @staticmethod
    def _parse_outpaint_padding(padding_spec: str):
        """
        Parse outpaint padding string like: l200r200t200b200.
        Missing sides default to 0.
        """
        matches = re.findall(r"([lrtb])(\d+)", padding_spec.lower())
        if not matches:
            raise ValueError("Invalid outpaint padding format. Example: l200r200t200b200")

        pads = {"l": 0, "r": 0, "t": 0, "b": 0}
        seen = set()
        for side, val in matches:
            if side in seen:
                raise ValueError(f"Duplicate side '{side}' in outpaint padding: {padding_spec}")
            seen.add(side)
            pads[side] = int(val)

        if all(v == 0 for v in pads.values()):
            raise ValueError("Outpaint padding must contain at least one non-zero side.")

        return pads["l"], pads["r"], pads["t"], pads["b"]

    def _preprocess(
        self,
        image,
        mask_image,
        outpaint_padding,
        height,
        width,
        device,
    ):
        """
        Preprocess image & mask.

        Returns
        -------
        orig_pil : PIL.Image
        mask_keep_pixel : Tensor (1, 1, H, W)
            Keep mask. 1 = keep, 0 = edit.
        img_tensor : Tensor
        image_height, image_width : int
        """
        from diffusers.utils import load_image

        model_dtype = self.adapter.dtype
        orig_img = load_image(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        img_w, img_h = orig_img.size
        use_generated_outpaint_mask = outpaint_padding is not None
        multiple_of = self.adapter.vae_scale_factor * 2

        if use_generated_outpaint_mask:
            if mask_image is not None:
                raise ValueError("Do not pass mask_image when outpaint_padding is set.")
            if height is not None or width is not None:
                raise ValueError("Do not pass height/width when outpaint_padding is set.")

            pad_l, pad_r, pad_t, pad_b = self._parse_outpaint_padding(outpaint_padding)

            # 1) Strict ComfyUI ImagePadForOutpaint logic.
            comfy_w = img_w + pad_l + pad_r
            comfy_h = img_h + pad_t + pad_b
            canvas = Image.new("RGB", (comfy_w, comfy_h), (127, 127, 127))
            canvas.paste(orig_img, (pad_l, pad_t))
            orig_img = canvas

            # Hard outpaint mask:
            # - padding region is edit (0 in keep-mask)
            # - original image region is keep (1 in keep-mask)
            # - additionally, if a side is padded, mask 20 px inward for seam redraw.
            mask_keep_np = np.zeros((comfy_h, comfy_w), dtype=np.float32)
            mask_keep_np[pad_t: pad_t + img_h, pad_l: pad_l + img_w] = 1.0
            inner_mask_px = 20
            if pad_l > 0:
                x0 = pad_l
                x1 = min(pad_l + inner_mask_px, pad_l + img_w)
                mask_keep_np[pad_t: pad_t + img_h, x0:x1] = 0.0
            if pad_r > 0:
                x0 = max(pad_l + img_w - inner_mask_px, pad_l)
                x1 = pad_l + img_w
                mask_keep_np[pad_t: pad_t + img_h, x0:x1] = 0.0
            if pad_t > 0:
                y0 = pad_t
                y1 = min(pad_t + inner_mask_px, pad_t + img_h)
                mask_keep_np[y0:y1, pad_l: pad_l + img_w] = 0.0
            if pad_b > 0:
                y0 = max(pad_t + img_h - inner_mask_px, pad_t)
                y1 = pad_t + img_h
                mask_keep_np[y0:y1, pad_l: pad_l + img_w] = 0.0

            # 2) Align to VAE grid by extending right/bottom only.
            iw = ((comfy_w + multiple_of - 1) // multiple_of) * multiple_of
            ih = ((comfy_h + multiple_of - 1) // multiple_of) * multiple_of
            if iw != comfy_w or ih != comfy_h:
                aligned_canvas = Image.new("RGB", (iw, ih), (127, 127, 127))
                aligned_canvas.paste(orig_img, (0, 0))
                orig_img = aligned_canvas

                aligned_keep = np.zeros((ih, iw), dtype=np.float32)
                aligned_keep[:comfy_h, :comfy_w] = mask_keep_np
                mask_keep_np = aligned_keep

            mask_keep = torch.from_numpy(mask_keep_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            if mask_image is None:
                raise ValueError("mask_image is required when outpaint_padding is not set.")

            mask_loaded = load_image_preserve_alpha(mask_image)
            mask_rgba = mask_loaded.convert("RGBA")
            _, _, _, alpha = mask_rgba.split()
            if np.array(alpha).min() < 250:
                mask_src = alpha
            else:
                mask_src = mask_loaded.convert("L")

            if height is None or width is None:
                max_area = 1024 * 1024
                iw, ih = img_w, img_h
                if iw * ih > max_area:
                    scale = (max_area / (iw * ih)) ** 0.5
                    iw = int(iw * scale)
                    ih = int(ih * scale)
                    orig_img = orig_img.resize((iw, ih), Image.BICUBIC)
                    mask_src = mask_src.resize((iw, ih), Image.Resampling.NEAREST)
            else:
                iw, ih = width, height
                mask_src = mask_src.resize((iw, ih), Image.Resampling.NEAREST)

            iw = (iw // multiple_of) * multiple_of
            ih = (ih // multiple_of) * multiple_of
            if orig_img.size != (iw, ih):
                orig_img = orig_img.resize((iw, ih), Image.BICUBIC)

            left = (mask_src.width - iw) // 2
            top = (mask_src.height - ih) // 2
            mask_cropped = mask_src.crop((left, top, left + iw, top + ih))
            mask_tensor = (
                torch.from_numpy(np.array(mask_cropped).astype(np.float32) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            mask_keep = (mask_tensor > 0.01).float()

        img_tensor = self.adapter.image_processor.preprocess(
            orig_img, height=ih, width=iw, resize_mode="crop"
        ).to(device, model_dtype)

        return orig_img, mask_keep, img_tensor, ih, iw

    def _decode(self, latents):
        return self.adapter.decode_latents(latents)

    @staticmethod
    def _save_preprocess_debug(
        out_dir: str,
        orig_img: Image.Image,
        img_tensor: torch.Tensor,
        mask_keep: torch.Tensor,
    ) -> None:
        """Save exact preprocessed inputs used by the network for inspection."""
        os.makedirs(out_dir, exist_ok=True)

        orig_img.save(os.path.join(out_dir, "preprocess_orig_canvas.png"))

        img_np = img_tensor[0].detach().float().cpu().permute(1, 2, 0).numpy()
        img_np = ((img_np / 2.0) + 0.5).clip(0.0, 1.0)
        Image.fromarray((img_np * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "preprocess_image_tensor_vis.png")
        )

        keep_np = mask_keep[0, 0].detach().float().cpu().numpy().clip(0.0, 1.0)
        Image.fromarray((keep_np * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "preprocess_mask_keep.png")
        )
        Image.fromarray(((1.0 - keep_np) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "preprocess_mask_edit.png")
        )
