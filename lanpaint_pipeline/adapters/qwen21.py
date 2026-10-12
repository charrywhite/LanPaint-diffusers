"""
Qwen21Adapter — adapts QwenImage21Pipeline (Qwen-Image 2.1) to LanPaint's interface.

Qwen-Image 2.1 works in RGBA, edits with condition images placed in its
sequence, and its transformer is block-causal. The adapter follows
QwenImage21Pipeline (diffusers 0.41) on each point:

- The canvas is <image1>; set_references() adds <image2>, <image3>, ... Each
  condition is resized once to an area of condition_resolution² (its own aspect
  ratio, multiples of 32), and that same image feeds the VL encoder and the VAE.
- The VL encoder sees the alpha composited over white (inside encode_prompt);
  the VAE reads all four channels. The negative prompt is encoded with the same
  conditions.
- The text and condition prefix does not depend on the timestep: its keys and
  values are extracted on the first call of a request and reused afterwards, one
  QwenImage21KVCache per CFG branch (use_kv_cache = False recomputes them).
- The schedule is the pipeline's: explicit sigmas, else the checkpoint's
  sample_sigmas, else linspace(1, 1/N, N), shifted for the target token count.

LanPaint keeps the latents of the whole canvas, unmasked, at the run size.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


# ---- pure helpers ----


def project_keep_mask(mask_pixel_keep: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    """
    Pixel keep-mask (1, 1, H, W) -> packed latent keep-mask (1, L, 1), 1 = keep.

    As ComfyUI LanPaint does: nearest-exact sampling to the latent grid, then a
    0.5 threshold. An edit thinner than a latent cell disappears when no sampled
    pixel falls on it.
    """
    if mask_pixel_keep.ndim != 4 or mask_pixel_keep.shape[:2] != (1, 1):
        raise ValueError(f"Expected one keep-mask of shape (1, 1, H, W), got {tuple(mask_pixel_keep.shape)}")
    keep = torch.nn.functional.interpolate(
        mask_pixel_keep.float(), size=(latent_height, latent_width), mode="nearest-exact"
    ) > 0.5
    return keep.float().reshape(1, -1, 1)


def schedule_sigmas(num_steps: int, explicit=None, checkpoint=None) -> np.ndarray:
    """Sampling sigmas without the final 0: explicit, else the checkpoint's, else linspace(1, 1/N, N)."""
    sigmas = explicit if explicit is not None else checkpoint
    if sigmas is None:
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}")
        return np.linspace(1.0, 1.0 / num_steps, num_steps)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if (sigmas.ndim != 1 or sigmas.size == 0 or not np.isfinite(sigmas).all()
            or (sigmas <= 0).any() or (sigmas > 1).any() or (np.diff(sigmas) >= 0).any()):
        raise ValueError("sigmas must decrease strictly within (0, 1], without the final 0")
    return sigmas


def model_timestep(sigma: float, dtype: torch.dtype, device=None) -> torch.Tensor:
    """
    The timestep the transformer receives, rounded as QwenImage21Pipeline rounds it:
    the scheduler's float32 timestep (sigma * 1000), cast to the latents' dtype, / 1000.
    """
    t = torch.tensor([sigma], dtype=torch.float32, device=device) * 1000
    return t.to(dtype) / 1000


def guided_x0(x, sigma, v_cond, v_uncond, guidance_scale: float, cfg_big: float):
    """
    LanPaint's two x0 estimates, x0 = x - sigma * v, in float32.

    The main estimate uses v_cond, or v_uncond + guidance_scale * (v_cond - v_uncond)
    above 1, as QwenImage21Pipeline does. The BIG estimate always uses cfg_big,
    which may be below 1 (ComfyUI's Prompt First is -0.5).
    """
    x, v_cond = x.float(), v_cond.float()
    if v_uncond is None:
        if guidance_scale > 1 or cfg_big != 1:
            raise ValueError("Guidance needs the unconditional velocity")
        x0 = x - sigma * v_cond
        return x0, x0
    v_uncond = v_uncond.float()
    main = v_cond if guidance_scale <= 1 else v_uncond + guidance_scale * (v_cond - v_uncond)
    big = v_uncond + cfg_big * (v_cond - v_uncond)
    return x - sigma * main, x - sigma * big


def check_condition_layout(image_pad_mask: torch.Tensor, condition_tokens: Sequence[int]) -> None:
    """
    The VL image slots form one run per condition, in order, and each run holds one
    slot per 2x2 group of that condition's VAE tokens. Otherwise the transformer
    would pair text positions with the wrong image tokens.
    """
    if image_pad_mask.dtype != torch.bool or image_pad_mask.ndim != 2 or image_pad_mask.shape[0] != 1:
        raise ValueError("image_pad_mask must be one boolean row")
    if any(tokens % 4 for tokens in condition_tokens):
        raise ValueError(f"Condition token counts must be multiples of 4, got {list(condition_tokens)}")
    slots = image_pad_mask[0].int().tolist()
    runs, length = [], 0
    for slot in slots + [0]:
        if slot:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    expected = [tokens // 4 for tokens in condition_tokens]
    if runs != expected:
        raise ValueError(f"Conditioning has image slot runs {runs}, the conditions need {expected}")


def append_target_slots(image_pad_mask: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """One image slot per 2x2 group of target latents, after the prompt, as the pipeline appends them."""
    if target_tokens % 4:
        raise ValueError(f"Target token count must be a multiple of 4, got {target_tokens}")
    return torch.cat([image_pad_mask, image_pad_mask.new_ones(image_pad_mask.shape[0], target_tokens // 4)], dim=1)


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """(1, C, H, W) in [-1, 1] -> 8-bit PIL image, exact for tensors made from 8-bit images."""
    array = ((img_tensor[0].detach().float().cpu() + 1) / 2 * 255).round().clamp(0, 255)
    return Image.fromarray(array.permute(1, 2, 0).numpy().astype(np.uint8))


class Qwen21Adapter(ModelAdapter):
    """
    Concrete adapter for QwenImage21Pipeline.

    Usage::

        pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16)
        adapter = Qwen21Adapter(pipe)
        adapter.set_references([Image.open("material.png")])  # optional <image2>
    """

    # Area of each condition image, as QwenImage21Pipeline's output_resolution.
    condition_resolution: int = 1024
    # Explicit sampling sigmas, without the final 0. None follows the checkpoint.
    sigmas: Optional[Sequence[float]] = None
    # Reuse the prefix keys and values across calls, as QwenImage21Pipeline's use_kv_cache.
    use_kv_cache: bool = True

    def __init__(self, pipe: QwenImage21Pipeline):
        super().__init__(pipe)
        self._references: Tuple[Image.Image, ...] = ()
        self._prompt: Optional[str] = None
        self._negative_prompt: str = ""
        # Raw encode_prompt outputs: (prompt_embeds, prompt_embeds_mask, image_pad_mask)
        self._positive = None
        self._negative = None
        self._conditioning_preloaded = False
        # Populated by encode_and_prepare, for one request
        self._conditions: List[Image.Image] = []
        self._condition_latents = None
        self._condition_tokens: List[int] = []
        self._img_shapes = None
        self._branches = {}
        self._caches = {}
        self._primed = {}
        self._latent_hw = (0, 0)
        self._pixel_hw = (0, 0)

    @property
    def requires_ref_at_inference(self) -> bool:
        return True

    @property
    def supports_alpha(self) -> bool:
        return True

    # ---- references ----

    @property
    def references(self) -> Tuple[Image.Image, ...]:
        """Condition images shown after the canvas, in order."""
        return self._references

    def set_references(self, images: Sequence[Image.Image]) -> None:
        """
        Show these images to the model after the canvas, as <image2>, <image3>, ...
        Replaces the previous list; an empty list leaves the canvas alone.
        Takes effect on the next run.
        """
        images = tuple(images)
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("References must be PIL images")
        self._references = tuple(image.copy() for image in images)

    # ---- conditioning ----

    def set_conditioning(self, positive_conditioning, negative_conditioning=None):
        """
        Inject conditioning encoded elsewhere, for the next run (called with prompt=None).

        Dicts with prompt_embeds, prompt_embeds_mask (optional) and image_pad_mask,
        as QwenImage21Pipeline.encode_prompt returns them for the same conditions
        (the canvas, then the references, resized as in encode_and_prepare).
        Replaces any earlier conditioning; without a negative, the empty prompt is
        encoded when guidance needs one.
        """
        def unpack(c):
            return c["prompt_embeds"], c.get("prompt_embeds_mask"), c["image_pad_mask"]

        self._positive = unpack(positive_conditioning)
        self._negative = None if negative_conditioning is None else unpack(negative_conditioning)
        self._negative_prompt = ""
        self._conditioning_preloaded = True
        self._prompt_bundle = PromptBundle(data=dict(positive_conditioning))

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        """
        Store the strings: the VL encoder needs the conditions, prepared in
        encode_and_prepare(). A text prompt replaces injected conditioning.
        """
        self._prompt = prompt
        self._negative_prompt = negative_prompt or ""
        self._conditioning_preloaded = False
        self._prompt_bundle = PromptBundle(data={"prompt": prompt, "negative_prompt": self._negative_prompt})
        return self._prompt_bundle

    def _encode(self, prompt: str, device: torch.device):
        embeds, mask, image_pad_mask = self.pipe.encode_prompt(prompt=prompt, image=self._conditions, device=device)
        return embeds, mask, image_pad_mask

    def _branch(self, raw):
        """Validate one branch's conditioning against the current conditions and add the target slots."""
        embeds, mask, image_pad_mask = raw
        check_condition_layout(image_pad_mask, self._condition_tokens)
        lat_h, lat_w = self._latent_hw
        return embeds, mask, append_target_slots(image_pad_mask, lat_h * lat_w)

    # ---- ModelAdapter implementation ----

    def encode_and_prepare(
        self,
        img_tensor: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> ImageLatents:
        """
        Prepare one request. img_tensor is the RGBA canvas at the run size, in [-1, 1].

        VAE-encode it at the run size (what LanPaint keeps); resize and VAE-encode the
        conditions (the canvas, then the references); VL-encode the prompt with them;
        start fresh prefix caches.
        """
        pipe = self.pipe
        dtype = self.dtype
        sf = pipe.vae_scale_factor
        self._pixel_hw = (height, width)
        self._latent_hw = (2 * (height // (sf * 2)), 2 * (width // (sf * 2)))
        lat_h, lat_w = self._latent_hw

        # One resize per condition, shared by the VL encoder and the VAE.
        conditions, vae_images, sizes = [], [], []
        for image in (_to_pil(img_tensor), *self._references):
            image = image if image.mode == "RGBA" else image.convert("RGBA")
            w, h, _ = calculate_dimensions(self.condition_resolution ** 2, image.width / image.height)
            resized = pipe.image_processor.resize(image, height=h, width=w)
            conditions.append(resized)
            vae_images.append(pipe.image_processor.preprocess(resized, height=h, width=w).unsqueeze(2))
            sizes.append((w, h))
        self._conditions = conditions
        self._condition_tokens = [(h // sf) * (w // sf) for w, h in sizes]
        self._img_shapes = [[*((1, h // sf, w // sf) for w, h in sizes), (1, lat_h, lat_w)]]

        # Passing latents keeps prepare_latents from drawing noise from the generator.
        channels = pipe.transformer.config.in_channels
        placeholder = torch.zeros(1, lat_h * lat_w, channels, device=device, dtype=dtype)
        _, self._condition_latents = pipe.prepare_latents(
            vae_images, 1, channels, height, width, dtype, device, generator, latents=placeholder
        )
        _, canvas_latents = pipe.prepare_latents(
            [img_tensor.unsqueeze(2)], 1, channels, height, width, dtype, device, generator, latents=placeholder
        )

        if not self._conditioning_preloaded:
            self._positive = self._encode(self._prompt, device)
            self._negative = None
        self._branches = {
            "cond": self._branch(self._positive),
            "uncond": None if self._negative is None else self._branch(self._negative),
        }

        cached = self.use_kv_cache and pipe.transformer.config.causal_condition
        blocks = len(pipe.transformer.transformer_blocks)
        self._caches = {b: QwenImage21KVCache(blocks) for b in ("cond", "uncond")} if cached else {}
        self._primed = {"cond": False, "uncond": False}

        self._image_latents = ImageLatents(
            latent=canvas_latents.float(),
            meta={"latent_height": lat_h, "latent_width": lat_w, "pixel_height": height, "pixel_width": width},
        )
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        return project_keep_mask(mask_pixel_keep, *self._latent_hw)

    def prepare_timesteps(self, num_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """QwenImage21Pipeline's schedule, shifted by mu for the target token count."""
        sigmas = schedule_sigmas(num_steps, self.sigmas, self.pipe.config.sample_sigmas)
        cfg = self.pipe.scheduler.config
        lat_h, lat_w = self._latent_hw
        mu = calculate_shift(
            lat_h * lat_w,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, len(sigmas), device, sigmas=sigmas, mu=mu)
        if not torch.isfinite(timesteps).all() or not torch.isfinite(self.pipe.scheduler.sigmas).all():
            raise ValueError(
                "The Qwen scheduler produced non-finite timesteps or sigmas. "
                "Use at least two inference steps with the default schedule, "
                "or supply a valid explicit schedule."
            )
        return timesteps, self.pipe.scheduler.sigmas.to(device)[:-1]

    def _velocity(self, x: torch.Tensor, sigma: float, branch: str) -> torch.Tensor:
        """The transformer called as QwenImage21Pipeline calls it; the target tokens' velocity in float32."""
        dtype = self.dtype
        embeds, mask, image_pad_mask = self._branches[branch]
        cache = self._caches.get(branch)
        mode = None if cache is None else ("cached" if self._primed[branch] else "extract")
        hidden = torch.cat([self._condition_latents, x.to(dtype)], dim=1)
        with self.pipe.transformer.cache_context(branch):
            v = self.pipe.transformer(
                hidden_states=hidden,
                timestep=model_timestep(sigma, dtype, x.device),
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=mask,
                img_shapes=self._img_shapes,
                img_mask=image_pad_mask,
                attention_kwargs={},
                kv_cache=cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0][:, -x.shape[1]:]
        if cache is not None:
            self._primed[branch] = True
        return v.float()

    def predict_x0(
        self,
        x: torch.Tensor,
        flow_t: float,
        guidance_scale: float,
        cfg_big: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x0 = x - t * v. The negative branch runs only when a guidance scale needs it."""
        v_cond = self._velocity(x, flow_t, "cond")
        v_uncond = None
        if guidance_scale > 1 or cfg_big != 1:
            if self._branches["uncond"] is None:
                self._branches["uncond"] = self._branch(self._encode(self._negative_prompt, x.device))
            v_uncond = self._velocity(x, flow_t, "uncond")
        return guided_x0(x, flow_t, v_cond, v_uncond, guidance_scale, cfg_big)

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Packed latents -> denormalize -> RGBA VAE decode -> RGBA PIL, as the pipeline decodes."""
        pipe = self.pipe
        height, width = self._pixel_hw
        with torch.no_grad():
            unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
            z = pipe.vae.config.z_dim
            mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z, 1, 1, 1).to(unpacked.device, unpacked.dtype)
            std = torch.tensor(pipe.vae.config.latents_std).view(1, z, 1, 1, 1).to(unpacked.device, unpacked.dtype)
            image = pipe.vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]
