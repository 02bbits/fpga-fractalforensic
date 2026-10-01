#!/usr/bin/env bash
# HyperReenact pretrained + auxiliary models (Google Drive).
set -euo pipefail
cd "$(dirname "$0")/../HyperReenact"

mkdir -p pretrained_models/data
uvx gdown 1cBwIFwq6cYIA5iR8tEvj6BIL7Ji7azIH -O pretrained_models/stylegan-voxceleb.pt
uvx gdown 1TRATaREBi4VCMITUZV0ZO2XFU3YZKGlQ -O pretrained_models/e4e-voxceleb.pt
uvx gdown 1BUp6S3Wf2SeM3a-b_7mkKZyaXAxKhLrI -O pretrained_models/hypernetwork.pt
uvx gdown 1IWqJUTAZCelAZrUzfU38zK_ZM25fK32S -O pretrained_models/s3fd-619a316812.pth
uvx gdown 1F3wrQALEOd1Vku8ArJ_Gn4T6U3IX7Pz7 -O pretrained_models/insight_face.pth

# DECA auxiliary package: data.tar.gz -> pretrained_models/data/
uvx gdown 1BHVJAEXscaXMj_p2rOsHYF_vaRRRHQbA -O deca_data.tar.gz
tar xzf deca_data.tar.gz -C pretrained_models
rm -f deca_data.tar.gz
