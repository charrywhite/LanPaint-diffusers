#!/usr/bin/env bash
# Unified LanPaint CLI examples
# Usage: uncomment one command block, or copy and customize it.

# List all registered models
# python run_lanpaint.py --list-models

# Flux2 Klein (URL example)
# python run_lanpaint.py --model flux-klein \
#     --prompt "change building's window light color to blue" \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Masked_Load_Me_in_Loader.png"

# Flux2 Klein (local paths)
# python run_lanpaint.py --model flux-klein \
#     --prompt "change window light to blue" \
#     --image path/to/image.png \
#     --mask path/to/mask.png

# SD3
# python run_lanpaint.py --model sd3 \
#     --lp-n-steps 5 \
#     --guidance-scale 5.5 \
#     --num-steps 30 \
#     --prompt "a bottle with a rainbow galaxy inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background" \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_9/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_9/Masked_Load_Me_in_Loader.png"

# Z-Image Turbo Inpaint
python run_lanpaint.py --model z-image \
    --lp-n-steps 5 \
    --lp-friction 15.0 \
    --lp-lambda 16 \
    --seed 42 \
    --guidance-scale 1.0 \
    --num-steps 9 \
    --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
    --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Original_No_Mask.png" \
    --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Masked_Load_Me_in_Loader.png"

# Z-Image Turbo Outpaint
# python run_lanpaint.py --model z-image \
#     --lp-n-steps 5 \
#     --lp-friction 15.0 \
#     --lp-lambda 16 \
#     --seed 42 \
#     --guidance-scale 1.0 \
#     --num-steps 15 \
#     --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_22/Original_No_Mask.png" \
#     --outpaint-pad "l200r200t200b200" \

