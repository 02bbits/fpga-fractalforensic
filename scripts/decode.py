"""
Core op (512px): recover the fractal watermark from an image and score it.
Nothing else. Ground truth is regenerated with the same knobs scripts/encode.py
used — keep them in sync (defaults match).

Crop logic mirrors encode.py: the saved crop box is used when present, else the
image is center-cropped (a 512x512 input passes through unchanged). No stretching.

Run from the project root:
    python3 scripts/decode.py                                      # results/watermarked_512.png
    python3 scripts/decode.py --image photo.jpg --box results/watermark_box.json
    python3 scripts/decode.py --image wm.png --d 4                 # custom encode knob

Output: results/recovered_watermark.npy (override with --out) + bit/patch accuracy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling script imports

import numpy as np
import torch
from PIL import Image

from core import FractalForensicsPipeline
from core.metrics import bit_accuracy, patch_accuracy
from encode import (IMG_SIZE, LATENT_CHANNELS, WEIGHTS_DIR, WM_DEFAULTS,
                    WTM_SIZE, center_box, crop_to_tensor, generate)


def recover(pipeline, img):
    """Per-entry bit probabilities (1, 4, wtm, wtm) from an image tensor."""
    with torch.no_grad():
        return pipeline.decode(img)


def load_crop(path, box_path):
    """Image -> 512x512 tensor: crop at the saved box, else center crop."""
    img = Image.open(path).convert('RGB')
    if os.path.isfile(box_path):
        with open(box_path) as f:
            box = tuple(json.load(f)['box'])
    else:
        box = center_box(img)
    return crop_to_tensor(img, box)


def main():
    parser = argparse.ArgumentParser(description='Recover a fractal watermark from an image')
    parser.add_argument('--image', default='results/watermarked_512.png')
    parser.add_argument('--box', default='results/watermark_box.json',
                        help='crop box saved by encode.py (optional)')
    parser.add_argument('--out', default='results/recovered_watermark.npy')
    parser.add_argument('--weights', default=WEIGHTS_DIR)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--r', type=int, default=WM_DEFAULTS['r'])
    parser.add_argument('--m', type=int, default=WM_DEFAULTS['m'])
    parser.add_argument('--o', type=int, default=WM_DEFAULTS['o'])
    parser.add_argument('--x0', type=float, default=WM_DEFAULTS['x0'])
    parser.add_argument('--a', type=float, default=WM_DEFAULTS['a'])
    parser.add_argument('--k', type=int, default=WM_DEFAULTS['k'])
    parser.add_argument('--d', type=int, default=WM_DEFAULTS['d'])
    args = parser.parse_args()

    pipeline = FractalForensicsPipeline.from_weights(
        args.weights, img_size=IMG_SIZE, wtm_size=WTM_SIZE,
        latent_channels=LATENT_CHANNELS, device=args.device)

    img = load_crop(args.image, args.box).to(pipeline.device)
    watermark = generate(pipeline, r=args.r, m=args.m, o=args.o,
                         x0=args.x0, a=args.a, k=args.k, d=args.d)
    watermark_rec = recover(pipeline, img)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.save(args.out, watermark_rec.cpu().numpy())
    print(f'bit acc {bit_accuracy(watermark, watermark_rec):.4f} '
          f'patch acc {patch_accuracy(watermark, watermark_rec):.4f}')
    print(f'recovered watermark -> {args.out}')


if __name__ == '__main__':
    main()
