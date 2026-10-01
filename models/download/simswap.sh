#!/usr/bin/env bash
# SimSwap weights: arcface + generator checkpoints (GitHub releases),
# and insightface antelope models for face cropping.
set -euo pipefail
cd "$(dirname "$0")/../SimSwap"

mkdir -p arcface_model
wget -c -P arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
wget -c https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
unzip -o checkpoints.zip -d checkpoints
rm -f checkpoints.zip

mkdir -p insightface_func/models
if [ ! -d insightface_func/models/antelope ]; then
    wget -c --no-check-certificate \
        "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" \
        -O antelope.zip \
        || echo "WARNING: OneDrive antelope download failed (link may be dead) — see models/README.md for the fallback."
    if [ -f antelope.zip ]; then
        unzip -o antelope.zip -d insightface_func/models/
        rm -f antelope.zip
    fi
fi
