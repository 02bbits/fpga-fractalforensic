#!/usr/bin/env bash
# UniFace pretrained swap and reenactment models (Google Drive).
set -euo pipefail
cd "$(dirname "$0")/../UniFace"

mkdir -p session/swap/checkpoints session/reenactment/checkpoints
uvx gdown 1H3DHfld_M5F940bZMuoT_bDkW3tJZvbq -O session/swap/checkpoints/500000.pt
uvx gdown 1Y-Sm-_HmvPSBwz16Ol8apwnBUCYU4_tH -O session/reenactment/checkpoints/1000000.pt
