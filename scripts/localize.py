"""
Localization overlay for a deepfaked image: red where the watermark was lost.

Prerequisite: the deepfake must have been applied to the *watermarked* image
(scripts/encode.py output) — localizing a deepfake of a clean image is
meaningless, there is nothing embedded to lose. Expected flow:
    encode.py --image images/tony.jpg        -> results/watermarked_512.png
    <deepfake it>                            -> e.g. images/deepfaked-tony.jpg
    localize.py --image images/deepfaked-tony.jpg

The saved crop box is reused; if the deepfake changed the image resolution the
box is scaled by the resolution ratio. A fully red overlay means the watermark
was destroyed everywhere (e.g. deepfake made from the clean source).

Run from the project root:
    python3 scripts/localize.py --image images/deepfaked-tony.jpg
    python3 scripts/localize.py --image df.jpg --box results/watermark_box.json --out results/df_loc.png

Output: red-blend overlay on the deepfaked crop + bit/patch accuracy numbers.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling script imports

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from core import FractalForensicsPipeline
from core.metrics import bit_accuracy, patch_accuracy, patch_error_map
from decode import recover
from encode import (IMG_SIZE, LATENT_CHANNELS, WEIGHTS_DIR, WM_DEFAULTS,
                    WTM_SIZE, crop_to_tensor, generate)
from pipeline import localization_overlay


def scaled_box(box, source_path, deepfake_size):
    """Box from encode.py, scaled if the deepfake changed the resolution."""
    if not (source_path and os.path.isfile(source_path)):
        return box
    src_w, src_h = Image.open(source_path).size
    sx, sy = deepfake_size[0] / src_w, deepfake_size[1] / src_h
    if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
        return box
    sb = (round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy))
    print(f'resolution changed {src_w}x{src_h} -> {deepfake_size[0]}x{deepfake_size[1]}; '
          f'box {tuple(box)} -> {sb}')
    return sb


def main():
    parser = argparse.ArgumentParser(description='Localize watermark loss in a deepfaked image')
    parser.add_argument('--image', required=True, help='deepfaked image (from a watermarked source)')
    parser.add_argument('--box', default='results/watermark_box.json',
                        help='crop box saved by encode.py')
    parser.add_argument('--out', default='results/localization_deepfake.png')
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

    with open(args.box) as f:
        record = json.load(f)
    img = Image.open(args.image).convert('RGB')
    box = scaled_box(tuple(record['box']), record.get('image'), img.size)

    img_t = crop_to_tensor(img, box).to(pipeline.device)
    watermark = generate(pipeline, r=args.r, m=args.m, o=args.o,
                         x0=args.x0, a=args.a, k=args.k, d=args.d)
    watermark_rec = recover(pipeline, img_t)
    print(f'bit acc {bit_accuracy(watermark, watermark_rec):.4f} '
          f'patch acc {patch_accuracy(watermark, watermark_rec):.4f}')

    overlay = localization_overlay(img_t, patch_error_map(watermark, watermark_rec))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    TF.to_pil_image(overlay.clamp(0, 1)).save(args.out)
    print(f'localization overlay -> {args.out}')
    print('all red = watermark destroyed everywhere (check the deepfake was made from the '
          'watermarked image)')


if __name__ == '__main__':
    main()
