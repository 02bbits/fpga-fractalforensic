"""
512px end-to-end flow, executed automatically:
    image (center 512px crop, or the face crop with --face) -> generate + embed
    -> benign attacks -> recover -> score -> localization overlay.

Run from the project root:
    python3 scripts/pipeline.py                            # synthetic random image
    python3 scripts/pipeline.py --image photo.jpg
    python3 scripts/pipeline.py --image photo.jpg --face   # crop the face region first

Output: per-attack bit/patch accuracy table + results/localization_512.png
(red overlay where entries were lost after a simulated local edit).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling script imports

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from attacks.manipulations import ATTACKS
from core import FractalForensicsPipeline
from core.metrics import bit_accuracy, patch_accuracy, patch_error_map
from decode import recover
from encode import (IMG_SIZE, LATENT_CHANNELS, WEIGHTS_DIR, WM_DEFAULTS,
                    WTM_SIZE, center_box, crop_to_tensor, embed, generate, psnr)

PATCH = IMG_SIZE // WTM_SIZE


def face_image(path):
    """512px face crop of a photo (largest Haar face, center fallback)."""
    from face_pipeline import detect_face_box
    img = Image.open(path).convert('RGB')
    return crop_to_tensor(img, detect_face_box(np.asarray(img)))


def tamper(imgs_wtm, imgs_clean, x0=192, y0=192, w=128, h=128):
    """Simulated local edit (paper §4.4): restore original pixels in a region,
    wiping the embedded entries there (128px = 4x4 of the 16x16 entries)."""
    out = imgs_wtm.clone()
    out[:, :, y0:y0 + h, x0:x0 + w] = imgs_clean[:, :, y0:y0 + h, x0:x0 + w]
    return out


def localization_overlay(imgs_tampered, error_map, patch_size=PATCH):
    """Blend a red overlay over the entries whose watermark was lost."""
    err = error_map[0].float()[None, None]
    err = torch.nn.functional.interpolate(err, scale_factor=patch_size, mode='nearest').squeeze()
    vis = (imgs_tampered[0].clamp(-1, 1) + 1) / 2
    red = torch.tensor([1.0, 0.0, 0.0], device=vis.device).view(3, 1, 1)
    return vis * (1 - err * 0.6) + red * err * 0.6


def main():
    parser = argparse.ArgumentParser(description='FractalForensics 512px auto-evaluation flow')
    parser.add_argument('--image', help='input image; default = synthetic random')
    parser.add_argument('--face', action='store_true',
                        help='crop the 512px face region first (requires --image)')
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

    if args.face and not args.image:
        parser.error('--face requires --image')

    pipeline = FractalForensicsPipeline.from_weights(
        args.weights, img_size=IMG_SIZE, wtm_size=WTM_SIZE,
        latent_channels=LATENT_CHANNELS, device=args.device)

    if args.face:
        img = face_image(args.image)
    elif args.image:
        pil = Image.open(args.image).convert('RGB')
        img = crop_to_tensor(pil, center_box(pil))
    else:
        img = torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 2 - 1
    img = img.to(pipeline.device)
    watermark = generate(pipeline, r=args.r, m=args.m, o=args.o,
                         x0=args.x0, a=args.a, k=args.k, d=args.d)
    img_wtm = embed(pipeline, img, watermark)
    print(f'input: {"face crop" if args.face else "image"} | embed PSNR {psnr(img, img_wtm):.2f} dB')

    print(f'\n{"attack":<16}{"bit acc":>10}{"patch acc":>12}')
    print('-' * 38)
    for name, attack in ATTACKS.items():
        watermark_rec = recover(pipeline, attack(img_wtm))
        print(f'{name:<16}{bit_accuracy(watermark, watermark_rec):>10.4f}'
              f'{patch_accuracy(watermark, watermark_rec):>12.4f}')

    img_tampered = tamper(img_wtm, img)
    watermark_rec = recover(pipeline, img_tampered)
    print(f'\ntamper region 128x128 px (4x4 of the {WTM_SIZE}x{WTM_SIZE} entries): '
          f'patch acc {patch_accuracy(watermark, watermark_rec):.4f}')

    overlay = localization_overlay(img_tampered, patch_error_map(watermark, watermark_rec))
    os.makedirs('results', exist_ok=True)
    TF.to_pil_image(overlay.clamp(0, 1)).save('results/localization_512.png')
    print('localization overlay -> results/localization_512.png')


if __name__ == '__main__':
    main()
