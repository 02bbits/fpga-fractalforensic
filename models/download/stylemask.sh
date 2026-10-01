#!/usr/bin/env bash
# StyleMask pretrained models (Google Drive): StyleGAN2-FFHQ-1024, e4e, StyleMask.
set -euo pipefail
cd "$(dirname "$0")/../StyleMask"

mkdir -p pretrained_models
uvx gdown 1I01HVu9UUyzAV7rNNnbzyFqPF0msebD7 -O pretrained_models/stylegan2-ffhq-config-f_1024.pt
uvx gdown 1DexTMA3QMRNwQ3Xhdojki8UAYuY3g0uu -O pretrained_models/e4e_ffhq_encode_1024.pt
uvx gdown 1_V_MnFB8rh5qrQ3zJk00fKXb81uHjIU8 -O pretrained_models/mask_network_1024.pt
