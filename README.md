# LanPaint-Diffusers

Training-free diffusion inpainting and outpainting with [LanPaint](https://github.com/scraed/LanPaint), built on [Hugging Face Diffusers](https://github.com/huggingface/diffusers).

## Features

- **Diffusers-native**: Uses the `diffusers` library only. No ComfyUI or other graph-based UI dependency.
- **Multi-model**: One CLI and one pipeline API for all supported backends. Currently supported:
  - **Flux2 Klein** (`flux-klein`)
  - **Z-Image Turbo** (`z-image`)
  - **Krea 2** (`krea2`)
  - **Stable Diffusion 3** (`sd3`)
  - **Qwen Image** (`qwen`) — thanks [@spartanz51](https://github.com/spartanz51)
  - **Qwen Image 2.1** (`qwen-image-2.1`): RGBA edits, with optional reference images
- **Extensible**: More LanPaint-supported models will be added over time; new backends are integrated via the adapter registry.

## Documentation

- [LanPaint Project Overview](https://scraed.github.io/scraedBlog/lanpaint/)
- [Partial Conditional Sampling with Langevin Dynamics](https://scraed.github.io/scraedBlog/lanpaint/conditional-sampling/)
- [Training-Free Local Editing for Images, Video, and Audio](https://scraed.github.io/scraedBlog/lanpaint/local-editing/)
---

## Usage

### Installation

- Python 3.10+ (3.12 tested), NVIDIA GPU and CUDA-compatible PyTorch recommended.
- **LanPaint** and **diffusers** are installed from GitHub (see `requirements.txt`); 

```bash
cd LanPaint-diffusers   # or your clone path
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a different PyTorch/CUDA build, install PyTorch first, then:

```bash
pip install -r requirements.txt --no-deps
```

Verify:

```bash
python -c "import torch, diffusers, LanPaint; print(torch.__version__, diffusers.__version__)"
python run_lanpaint.py --list-models
```

### Commands

**We recommend using the example commands in `run_lanpaint.sh`.** Uncomment the block you need (or run it as-is if one is already uncommented), then:

```bash
./run_lanpaint.sh
# or: bash run_lanpaint.sh
```

The script includes ready-to-run examples for:

- **List models**: `python run_lanpaint.py --list-models`
- **Flux2 Klein** (inpaint, URL or local image + mask)
- **SD3** (inpaint with example prompt and URLs)
- **Z-Image Turbo** (inpaint and outpaint, with `--outpaint-pad`)
- **Krea 2 Turbo** (inpaint)
- **Qwen Image** (inpaint )

**Quick reference** (same CLI, custom args):

- Inpaint: `--model <name> --prompt "..." --image <path-or-URL> --mask <path-or-URL>`
- Outpaint: `--model <name> --prompt "..." --image <path> --outpaint-pad l200r200t200b200` (no `--mask`; do not use `--height`/`--width` with `--outpaint-pad`)

**Useful options**: `--guidance-scale`, `--num-steps`, `--seed`, `--output <path>`, `--model-id <hf-or-local-path>`, `--save-preprocess-dir <dir>`, `--local-files-only`

**Qwen Image 2.1**: the image is `<image1>` in the prompt; `--reference-image <path-or-URL>`, repeatable, adds `<image2>`, `<image3>`, ... in order (for example a material to borrow from). The model works in RGBA: a transparent image keeps its transparency, the result is RGBA, and the edit may change the outline. Outpainting is not available for this model.

```bash
python run_lanpaint.py --model qwen-image-2.1 \
    --prompt "The earcups of the headphones in <image1> are rebuilt with the material from <image2>" \
    --image canvas.png --mask keep-mask.png --reference-image material.png \
    --num-steps 20 --guidance-scale 4 --seed 777 \
    --lp-n-steps 5 --lp-lambda 5 --lp-step-size 0.2 --lp-min-step-frac 1 --lp-early-stop 1 --lp-cfg-big 1
```

These are the sampler settings of LanPaint's ComfyUI workflow for this model; Diffusers keeps its own noise schedule. `--lp-min-step-frac` defaults to 0 unless explicitly set. The mask reaches the model's 16-pixel latent grid by nearest sampling, so a stroke thinner than one cell can disappear. In the final composite, kept pixels within `--lp-blend-overlap // 2` pixels of the canvas border can change.

### Mask convention (`--mask`)

The pipeline supports both standalone mask images and RGBA masks:

- **Binary/grayscale mask image** (`L` mode or RGB converted to grayscale):
  - **0** (black, or normalized **0.0**) = region to be edited (inpaint area)
  - **255** (white, or normalized **1.0**) = region to keep (non-editable)
- **RGBA mask image** (mask stored in alpha channel):
  - **alpha = 0** = region to be edited (inpaint area)
  - **alpha = 255** = region to keep (non-editable)

The mask is always the separate `--mask` file, never the image's own alpha. It is resized with the image to the run size: `--height` and `--width` when both are given, otherwise the image size with its area capped at 1024² pixels; either way aligned down to the model's grid.
---

## Results Showcase

Below are visual examples for `flux-klein`, `z-image`, `krea2`, `qwen`, and `qwen-image-2.1`.

### Flux2 Klein (Example 24)

| Original | Masked | Diffusers Version | ComfyUI Version |
|:--------:|:------:|:-----------------:|:---------------:|
| ![Original Flux2 Klein](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Original_No_Mask.png) | ![Masked Flux2 Klein](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Masked_Load_Me_in_Loader.png) | ![Inpainted Flux2 Klein Diffusers](results/flux-klein/lanpaint_output.png) | ![Inpainted Flux2 Klein ComfyUI](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/InPainted_Drag_Me_to_ComfyUI.png) |

### Z-Image Turbo (Example 21)

| Original | Masked | Diffusers Version | ComfyUI Version |
|:--------:|:------:|:-----------------:|:---------------:|
| ![Original Z-Image](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Original_No_Mask.png) | ![Masked Z-Image](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Masked_Load_Me_in_Loader.png) | ![Inpainted Z-Image Diffusers](results/z-image/lanpaint_output.png) | ![Inpainted Z-Image ComfyUI](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/InPainted_Drag_Me_to_ComfyUI.png) |

### Krea 2 Turbo (Example 28)

| Original | Masked | Diffusers Version | ComfyUI Version |
|:--------:|:------:|:-----------------:|:---------------:|
| ![Original Krea 2](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_28/Original_No_Mask.png) | ![Masked Krea 2](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_28/Masked_Load_Me_in_Loader.png) | ![Inpainted Krea 2 Diffusers](results/krea2/lanpaint_output.png) | ![Inpainted Krea 2 ComfyUI](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_28/InPainted_Drag_Me_to_ComfyUI.png) |

### Qwen Image Edit (Example 14)

| Original | Masked | Diffusers Version | ComfyUI Version |
|:--------:|:------:|:-----------------:|:---------------:|
| ![Original Qwen](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Original_No_Mask.png) | ![Masked Qwen](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Masked_Load_Me_in_Loader.png) | ![Inpainted Qwen Diffusers](results/qwen/lanpaint_output.png) | ![Inpainted Qwen ComfyUI](https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/QwenEdit_2509_InPainted_Drag_Me_to_ComfyUI.png) |

### Qwen Image 2.1 (Example 32)

Masked RGBA editing with a material reference, based on
[LanPaint Example 32](https://github.com/scraed/LanPaint/tree/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32).
Settings: 20 steps, CFG 4, seed 777.

| Source | Keep mask (white = keep) | Material reference |
|:------:|:------------------------:|:------------------:|
| ![Source](https://raw.githubusercontent.com/scraed/LanPaint/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32/Image_Load_Me_in_Loader.png) | ![Keep mask](results/qwen-image-2.1/keep-mask.png) | ![Material](https://raw.githubusercontent.com/scraed/LanPaint/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32/Material_Load_Me_in_Loader.png) |

| Diffusers | ComfyUI |
|:---------:|:-------:|
| ![Diffusers result](results/qwen-image-2.1/lanpaint_output.png) | ![ComfyUI result](https://raw.githubusercontent.com/scraed/LanPaint/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32/InPainted_Drag_Me_to_ComfyUI.png) |

Run this example:

```bash
python run_lanpaint.py --model qwen-image-2.1 --device cuda \
    --prompt "$(cat results/qwen-image-2.1/prompt.txt)" \
    --image https://raw.githubusercontent.com/scraed/LanPaint/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32/Image_Load_Me_in_Loader.png \
    --mask results/qwen-image-2.1/keep-mask.png \
    --reference-image https://raw.githubusercontent.com/scraed/LanPaint/2d7912f9a5efe5ece8de334c7ca18317b8288c39/examples/Example_32/Material_Load_Me_in_Loader.png \
    --num-steps 20 --guidance-scale 4 --seed 777 \
    --lp-n-steps 5 --lp-lambda 5 --lp-step-size 0.2 \
    --lp-min-step-frac 1 --lp-early-stop 1 --lp-cfg-big 1 \
    --lp-blend-overlap 9 --output results/qwen-image-2.1/example.png
```

The checkpoint uses the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE), which requires a separate license for commercial use.

---

## Library Structure

```
LanPaint-diffusers/
├── README.md
├── requirements.txt
├── run_lanpaint.py      # Unified CLI
├── run_lanpaint.sh      # Example commands
└── lanpaint_pipeline/
    ├── __init__.py
    ├── model_adapter.py   # Abstract adapter interface
    ├── pipeline.py        # LanPaintInpaintPipeline (orchestrator)
    ├── registry.py        # Model registry (flux-klein, sd3, z-image, krea2, qwen, qwen-image-2.1)
    ├── utils.py           # Blend, time helpers, image loading
    └── adapters/
        ├── __init__.py
        ├── flux_klein.py  # Flux2KleinAdapter
        ├── krea2.py       # Krea2Adapter
        ├── qwen.py        # QwenAdapter
        ├── qwen21.py      # Qwen21Adapter
        ├── sd3.py         # SD3Adapter
        └── z_image.py     # ZImageAdapter
```

- **LanPaintInpaintPipeline** (`pipeline.py`): model-agnostic flow (preprocess → encode → LanPaint Langevin + scheduler loop → decode → blend).
- **ModelAdapter** (`model_adapter.py`): abstract interface (encode_prompt, encode_and_prepare, mask_to_latent_space, prepare_timesteps, predict_x0, decode_latents). Each backend implements one adapter in `adapters/`.
- **Registry** (`registry.py`): maps model keys to pipeline class, adapter class, and default config; adding a model is done by registering one entry.

---

## To-Do

- [ ] **Edit-model reference image switch**: Add an option to control whether edit-style models (e.g. Flux Klein, Qwen) use the reference image as input at inference.

---

## Citation

If you use this repository or LanPaint in your work, please cite:

```bibtex
@article{
zheng2025lanpaint,
title={LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling},
author={Candi Zheng and Yuan Lan and Yang Wang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=JPC8JyOUSW},
note={}
}
```
