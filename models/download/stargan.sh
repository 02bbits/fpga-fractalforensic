#!/usr/bin/env bash
# StarGAN v2 CelebA-HQ pretrained network + latent mean (Dropbox, dl=1).
set -euo pipefail
cd "$(dirname "$0")/../stargan-v2"

mkdir -p expr/checkpoints/celeba_hq
wget -c "https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=1" \
    -O expr/checkpoints/celeba_hq/100000_nets_ema.ckpt
wget -c "https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt?dl=1" \
    -O expr/checkpoints/wing.ckpt
wget -c "https://www.dropbox.com/s/91fth49gyb7xksk/celeba_lm_mean.npz?dl=1" \
    -O expr/checkpoints/celeba_lm_mean.npz
