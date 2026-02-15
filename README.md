# LanPaint Release Package

This folder is a standalone package for multi-model LanPaint inpainting/outpainting.

It includes:

- `lanpaint_pipeline/`: model-agnostic orchestration + model adapters
- `run_lanpaint.py`: unified CLI entrypoint
- `run_lanpaint.sh`: runnable command examples

## Folder Structure

```text
lanpaint_release/
├── README.md
├── run_lanpaint.py
├── run_lanpaint.sh
└── lanpaint_pipeline/
    ├── __init__.py
    ├── model_adapter.py
    ├── pipeline.py
    ├── registry.py
    ├── utils.py
    └── adapters/
        ├── __init__.py
        ├── flux_klein.py
        ├── sd3.py
        └── z_image.py
```

## What This Package Does

The package wraps multiple diffusion pipelines behind one consistent LanPaint workflow:

1. Load and preprocess image/mask
2. Encode prompt and image latents via adapter
3. Run LanPaint Langevin dynamics in latent space
4. Run scheduler denoising loop
5. Decode latents to image
6. Blend generated and original image with a smooth mask

The architecture separates generic logic from model-specific logic:

- `LanPaintInpaintPipeline` (`pipeline.py`): shared orchestration
- `ModelAdapter` (`model_adapter.py`): abstract interface
- Adapter implementations (`adapters/*.py`): per-model details

## Supported Models

Model keys are defined in `lanpaint_pipeline/registry.py`:

- `flux-klein`
- `sd3`
- `z-image`

You can list models at runtime:

```bash
python run_lanpaint.py --list-models
```

## Requirements

This release includes `requirements.txt` with tested versions. The **LanPaint** library (used for `from LanPaint.lanpaint import LanPaint`) is installed from GitHub: [scraed/LanPaint](https://github.com/scraed/LanPaint) — it is not published on PyPI.

Minimum assumptions:

- Python 3.10+ (3.12 tested)
- NVIDIA GPU + CUDA-compatible PyTorch for practical speed
- Internet access for downloading model weights (unless using local checkpoints)

## Minimal Setup

### 1) Create and activate a virtual environment

```bash
cd lanpaint_release
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If your machine needs a different PyTorch build (for example a specific CUDA version),
install PyTorch first from the official index, then run:

```bash
pip install -r requirements.txt --no-deps
```

### 3) Verify installation

```bash
python -c "import torch, diffusers, LanPaint; print(torch.__version__, diffusers.__version__)"
python run_lanpaint.py --list-models
```

You should see registered models such as `flux-klein`, `sd3`, and `z-image`.

## Quick Start

### 1) Inpaint with an explicit mask

```bash
python run_lanpaint.py \
  --model z-image \
  --prompt "Change the shirt color to blue" \
  --image path/to/image.png \
  --mask path/to/mask.png
```

### 2) Outpaint with padding spec

Use `--outpaint-pad` instead of `--mask`.
Format is side+pixels, for example:

- `l200r200` (expand left and right by 200 px)
- `l200r200t200b200` (expand all four sides by 200 px)

```bash
python run_lanpaint.py \
  --model z-image \
  --prompt "Extend the scene naturally" \
  --image path/to/image.png \
  --outpaint-pad l200r200t200b200
```

Outpaint mode rules:

- `--outpaint-pad` and `--mask` are mutually exclusive
- when `--outpaint-pad` is used, do not pass `--height/--width`

## Helpful CLI Options

- `--guidance-scale`: CFG scale override
- `--num-steps`: scheduler step count override
- `--seed`: deterministic sampling seed
- `--save-preprocess-dir <dir>`: save debug images used before denoising
- `--output <path>`: custom output path
- `--model-id <hf-or-local-path>`: override model checkpoint
- `--local-files-only`: avoid Hub download

## Debug Artifacts

If you pass `--save-preprocess-dir`, the pipeline saves:

- `preprocess_orig_canvas.png`
- `preprocess_image_tensor_vis.png`
- `preprocess_mask_keep.png`
- `preprocess_mask_edit.png`
- `pre_blend_decoded_image.png`

These files are useful for diagnosing mask logic, outpaint boundaries, and blend behavior.

## Example Script

Use `run_lanpaint.sh` as a template for:

- model listing
- Flux2 Klein examples
- SD3 example
- Z-Image inpaint example
- Z-Image outpaint example

## Publishing to GitHub

To publish only this package:

1. Copy `lanpaint_release/` to your new repository root
2. Commit files
3. Push to GitHub

Recommended additions for a public repo:

- `LICENSE`
- `requirements.txt` or `environment.yml`
- a short `examples/` folder with sample input/output

