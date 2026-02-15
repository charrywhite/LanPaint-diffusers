"""
Unified LanPaint inpainting entry point — one script for ALL models.

Usage:
    # Flux2 Klein
    python run_lanpaint.py --model flux-klein \
        --prompt "change building's window light color to blue" \
        --image path/to/image.png \
        --mask path/to/mask.png

    # SD3
    python run_lanpaint.py --model sd3 \
        --prompt "A large panda bear walking through a stream of water." \
        --image path/to/image.png \
        --mask path/to/mask.png

    # List available models
    python run_lanpaint.py --list-models

    # Override model checkpoint
    python run_lanpaint.py --model sd3 --model-id my-org/my-sd3-finetune ...

Adding a new model: register it in lanpaint_pipeline/registry.py (one entry).
No new scripts needed.
"""

import argparse
import os
import sys

import torch

from lanpaint_pipeline import LanPaintConfig, LanPaintInpaintPipeline
from lanpaint_pipeline.registry import (
    create_adapter,
    get_model_spec,
    list_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LanPaint inpainting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument("--model", type=str, default="flux-klein",
                        help="Model key (e.g. flux-klein, sd3). Use --list-models to see all.")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override HuggingFace model ID or path to local checkpoint")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load from local path only (no Hub fetch); use with --model-id /path/to/checkpoint")
    parser.add_argument("--low-cpu-mem-usage", action="store_true",
                        help="Enable low CPU memory when loading (Z-Image local uses low_cpu_mem_usage=False by default)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    # I/O
    parser.add_argument("--prompt", type=str, required="--list-models" not in sys.argv,
                        help="Text prompt describing the edit")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--image", type=str, required="--list-models" not in sys.argv,
                        help="Path or URL to the source image")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path or URL to the mask image (alpha or grayscale, white=keep)")
    parser.add_argument("--outpaint-pad", type=str, default=None,
                        help="Generate outpaint mask from pad spec, e.g. l200r200 or l200r200t200b200")
    parser.add_argument("--save-preprocess-dir", type=str, default=None,
                        help="Save preprocessed network inputs (image/mask) to this directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: results/<model>/lanpaint_output.png)")

    # Generation parameters
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale (default: model-specific)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of inference steps (default: model-specific)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # LanPaint hyper-parameters
    parser.add_argument("--lp-n-steps", type=int, default=2,
                        help="LanPaint Langevin steps per scheduler step")
    parser.add_argument("--lp-friction", type=float, default=15.0)
    parser.add_argument("--lp-lambda", type=float, default=8.0)
    parser.add_argument("--lp-beta", type=float, default=1.0)
    parser.add_argument("--lp-step-size", type=float, default=0.2)
    parser.add_argument("--lp-early-stop", type=int, default=1)
    parser.add_argument("--lp-cfg-big", type=float, default=1.0)
    parser.add_argument("--lp-blend-overlap", type=int, default=9)

    return parser.parse_args()


def main():
    args = parse_args()

    # List models and exit
    if args.list_models:
        print("Available models:")
        for key in list_models():
            spec = get_model_spec(key)
            print(f"  {key:20s}  {spec.name}  ({spec.default_model_id})")
        return

    # Input validation: either normal inpaint (--mask) or outpaint (--outpaint-pad)
    if args.outpaint_pad:
        if args.mask is not None:
            raise ValueError("Do not pass --mask when --outpaint-pad is set.")
    else:
        if args.mask is None:
            raise ValueError("Missing --mask. For outpaint, use --outpaint-pad instead.")

    # Resolve model-specific defaults
    spec = get_model_spec(args.model)
    guidance_scale = args.guidance_scale or spec.default_params.get("guidance_scale", 5.0)
    num_steps = args.num_steps or spec.default_params.get("num_inference_steps", 20)

    # Create adapter (loads model)
    model_id = args.model_id
    from_pretrained_kwargs = {}

    # Z-Image: prefer local checkpoint if present (same as zimage_lanpaint_minimal.py)
    if args.model == "z-image" and model_id is None:
        local_zimage = os.path.join(os.getcwd(), "checkpoints", "Z-Image-Turbo")
        if os.path.isdir(local_zimage):
            model_id = local_zimage
            from_pretrained_kwargs["local_files_only"] = True
            from_pretrained_kwargs["low_cpu_mem_usage"] = False
            print(f"Using local Z-Image: {model_id}")

    if args.local_files_only:
        from_pretrained_kwargs["local_files_only"] = True
    if args.low_cpu_mem_usage:
        from_pretrained_kwargs["low_cpu_mem_usage"] = True

    effective_id = model_id or spec.default_model_id
    print(f"Loading model: {spec.name} ({effective_id})")
    adapter = create_adapter(
        args.model,
        device=args.device,
        model_id=model_id,
        **from_pretrained_kwargs,
    )

    # Build LanPaint config
    lp_config = LanPaintConfig(
        n_steps=args.lp_n_steps,
        friction=args.lp_friction,
        chara_lambda=args.lp_lambda,
        beta=args.lp_beta,
        step_size=args.lp_step_size,
        early_stop=args.lp_early_stop,
        cfg_big=args.lp_cfg_big,
        blend_overlap=args.lp_blend_overlap,
    )

    # Build pipeline and run
    lp_pipe = LanPaintInpaintPipeline(adapter, config=lp_config)

    result = lp_pipe(
        prompt=args.prompt,
        image=args.image,
        mask_image=args.mask,
        outpaint_padding=args.outpaint_pad,
        save_preprocess_dir=args.save_preprocess_dir,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        seed=args.seed,
    )

    # Save output
    output_path = args.output or f"results/{args.model}/lanpaint_output.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
