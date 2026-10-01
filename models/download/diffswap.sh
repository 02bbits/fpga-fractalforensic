#!/usr/bin/env bash
# DiffSwap checkpoints (Tsinghua cloud share):
#   diffswap.pth (~3.5 GB), glint360k_r100.pth (~261 MB),
#   shape_predictor_68_face_landmarks.dat (~100 MB).
set -euo pipefail
cd "$(dirname "$0")/../DiffSwap"

mkdir -p checkpoints
BASE="https://cloud.tsinghua.edu.cn/d/962ccd4b243442a3a144/files/?dl=1"
wget -c "$BASE&p=/checkpoints/DiffSwap/diffswap.pth" -O checkpoints/diffswap.pth
wget -c "$BASE&p=/checkpoints/DiffSwap/glint360k_r100.pth" -O checkpoints/glint360k_r100.pth
wget -c "$BASE&p=/checkpoints/DiffSwap/shape_predictor_68_face_landmarks.dat" -O checkpoints/shape_predictor_68_face_landmarks.dat
