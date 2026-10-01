#!/usr/bin/env bash
# E4S: RGI model + face parser (Google Drive), GPEN weights (direct URLs).
# facevid2vid stays manual (MediaFire) — see models/README.md.
set -euo pipefail
cd "$(dirname "$0")/../e4s"

mkdir -p pretrained_ckpts/e4s pretrained_ckpts/face_parsing
uvx gdown 1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq -O pretrained_ckpts/e4s/iteration_300000.pt
uvx gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O pretrained_ckpts/face_parsing/79999_iter.pth

(cd pretrained_ckpts/gpen && sh fetch_gepn_models.sh)

echo "MANUAL: facevid2vid 00000189-checkpoint.pth.tar (MediaFire, see INSTALLATION.md)" \
     "-> pretrained_ckpts/facevid2vid/"
