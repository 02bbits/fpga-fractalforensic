"""
Face-photo pipeline: crop the 512px face region, run the FractalForensics
pipeline on it, paste the watermarked crop back into the picture, then
recover the watermark from the pasted image and judge it with the
discriminator.

Run from the project root:
    python3 scripts/face_pipeline.py                    # hardcoded IMAGE_PATH below
    python3 scripts/face_pipeline.py --image photo.jpg  # any picture
    python3 scripts/face_pipeline.py --stage encode     # embed + paste only
    python3 scripts/face_pipeline.py --stage decode     # recover from saved output

Hardcoded input: IMAGE_PATH (edit below). Uses the 512px pretrained checkpoints.
Outputs in results/:
    face_watermarked_full.png   original photo with the watermarked face pasted back
    face_watermarked_crop.png   the 512px watermarked face crop
    face_box.json               face box used (x, y, w, h) + source image

The discriminator has no released checkpoint (training-only, random init) —
its logits are printed for inspection, not meaningful yet. See AGENTS.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling script imports

import cv2
import numpy as np
import torch
from PIL import Image

from core import FractalForensicsPipeline
from core.metrics import bit_accuracy, patch_accuracy
from decode import recover
from encode import (IMG_SIZE, LATENT_CHANNELS, WEIGHTS_DIR, WTM_SIZE,
                    crop_to_tensor, embed, generate, psnr, to_pil)

IMAGE_PATH = 'models/SimSwap/demo_file/Iron_man.jpg'   # <- hardcode your picture here
RESULTS_DIR = 'results'
FULL_OUT = os.path.join(RESULTS_DIR, 'face_watermarked_full.png')
CROP_OUT = os.path.join(RESULTS_DIR, 'face_watermarked_crop.png')
BOX_OUT = os.path.join(RESULTS_DIR, 'face_box.json')


def detect_face_box(rgb):
    """Largest Haar face, expanded 1.3x to a square box clipped to the image.

    Fallback (no detection): centered square of the shorter side.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
    if len(faces) == 0:
        print('no face detected -> center square fallback')
        s = min(h, w)
        return (w - s) // 2, (h - s) // 2, s, s
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    s = min(int(max(fw, fh) * 1.3), min(h, w))
    cx, cy = x + fw / 2, y + fh / 2
    x0 = max(0, min(int(cx - s / 2), w - s))
    y0 = max(0, min(int(cy - s / 2), h - s))
    return x0, y0, s, s


def disc_logits(pipeline, *imgs):
    """Discriminator outputs for comparison (random init: no released weights)."""
    with torch.no_grad():
        return [round(pipeline.discriminate(i.to(pipeline.device)).item(), 3) for i in imgs]


def encode(pipeline, image_path):
    """Crop the 512px face region, embed, paste the watermark back."""
    img = Image.open(image_path).convert('RGB')
    box = detect_face_box(np.asarray(img))
    clean = crop_to_tensor(img, box).to(pipeline.device)

    watermark = generate(pipeline)
    img_wtm = embed(pipeline, clean, watermark)
    print(f'face box (x, y, w, h): {box}')
    print(f'embed PSNR (clean vs watermarked crop): {psnr(clean, img_wtm):.2f} dB')
    print(f'discriminator logits (untrained): clean {disc_logits(pipeline, clean)[0]:+.3f} | '
          f'watermarked {disc_logits(pipeline, img_wtm)[0]:+.3f}')

    wtm_pil = to_pil(img_wtm)
    full = img.copy()
    full.paste(wtm_pil.resize((box[2], box[3])), (box[0], box[1]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    full.save(FULL_OUT)
    wtm_pil.save(CROP_OUT)
    with open(BOX_OUT, 'w') as f:
        json.dump({'box': list(box), 'image': image_path}, f)
    print(f'saved {FULL_OUT}, {CROP_OUT}, {BOX_OUT}')


def decode(pipeline, image_path):
    """Recover the watermark from the face region of the pasted-back image."""
    with open(BOX_OUT) as f:
        box = tuple(json.load(f)['box'])
    received = crop_to_tensor(Image.open(FULL_OUT).convert('RGB'), box).to(pipeline.device)

    watermark = generate(pipeline)
    watermark_rec = recover(pipeline, received)
    print(f'recovered from {FULL_OUT}: bit acc {bit_accuracy(watermark, watermark_rec):.4f} '
          f'patch acc {patch_accuracy(watermark, watermark_rec):.4f}')

    received_logit = disc_logits(pipeline, received)[0]
    print(f'discriminator logit on received crop (untrained): {received_logit:+.3f}')
    if os.path.isfile(image_path):
        clean = crop_to_tensor(Image.open(image_path).convert('RGB'), box).to(pipeline.device)
        print(f'discriminator logit on original crop (untrained): {disc_logits(pipeline, clean)[0]:+.3f}')


def main():
    parser = argparse.ArgumentParser(description='FractalForensics face-photo pipeline')
    parser.add_argument('--image', default=IMAGE_PATH,
                        help='input picture; default = hardcoded IMAGE_PATH')
    parser.add_argument('--stage', default='both', choices=['both', 'encode', 'decode'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    pipeline = FractalForensicsPipeline.from_weights(
        WEIGHTS_DIR, img_size=IMG_SIZE, wtm_size=WTM_SIZE,
        latent_channels=LATENT_CHANNELS, device=args.device)

    if args.stage in ('both', 'encode'):
        print('== encode: crop 512px face -> embed -> paste back ==')
        encode(pipeline, args.image)
    if args.stage in ('both', 'decode'):
        print('== decode: recover watermark from the pasted image ==')
        decode(pipeline, args.image)


if __name__ == '__main__':
    main()
