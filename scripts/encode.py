"""
Core op (512px): generate a fractal watermark, embed it into a 512x512 crop
of the image, and paste that crop back — no stretching, no attack/scoring
logic (flows live in pipeline.py / face_pipeline.py).

Run from the project root:
    python3 scripts/encode.py --image images/tony.jpg
    python3 scripts/encode.py --image photo.jpg --out results/wm.png

Crop = largest centered square, capped at 512px (upscaled if the source is
smaller). Outputs:
    results/watermarked_512.png     source image with the watermarked crop pasted back
    results/watermark_box.json      crop box used, so decode.py re-crops identically
The watermark knobs must match the ones passed to scripts/decode.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from core import FractalForensicsPipeline

IMG_SIZE = 512
WTM_SIZE = 16
LATENT_CHANNELS = 128
N = 4                       # watermark grid = 2^N x 2^N = 16x16 for the 512 model
WM_DEFAULTS = dict(r=1, m=2, o=3, x0=0.5, a=3.9, k=100, d=3)


def resolve_weights(size=IMG_SIZE):
    """Preferred checkpoint dir: root ``weights/<size>``, else the vendored copy.

    Both layouts exist while root weights/ is being populated (run from the
    project root).
    """
    root = f'weights/{size}'
    return root if os.path.isdir(root) else f'FractalForensics/weights/{size}'


WEIGHTS_DIR = resolve_weights()


def center_box(img):
    """Largest centered square, capped at 512px: (x, y, w, h)."""
    w, h = img.size
    s = min(IMG_SIZE, w, h)
    return (w - s) // 2, (h - s) // 2, s, s


def crop_to_tensor(img, box):
    """PIL crop at box -> (1, 3, 512, 512) tensor in [-1, 1] (resized to 512)."""
    crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3])).resize((IMG_SIZE, IMG_SIZE))
    return (TF.to_tensor(crop) * 2 - 1).unsqueeze(0)


def to_pil(tensor):
    """(1, 3, H, W) or (3, H, W) tensor in [-1, 1] -> PIL image."""
    return TF.to_pil_image(((tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1))


def paste_back(img, img_wtm, box):
    """Paste the watermarked 512 crop back into a copy of the source image."""
    full = img.copy()
    full.paste(to_pil(img_wtm).resize((box[2], box[3])), (box[0], box[1]))
    return full


def psnr(a, b):
    """PSNR between two [-1, 1] tensors (peak-to-peak = 2)."""
    mse = float(((a - b) ** 2).mean())
    return float('inf') if mse == 0 else 10 * float(torch.log10(torch.tensor(4.0 / mse)))


def generate(pipeline, **wm_params):
    """Watermark tensor (1, 4, 16, 16) for this resolution; knobs override WM_DEFAULTS."""
    return pipeline.generate_watermark(n=N, **{**WM_DEFAULTS, **wm_params})


def embed(pipeline, img, watermark):
    """Apply an already-generated watermark: image tensor -> watermarked tensor."""
    with torch.no_grad():
        return pipeline.embed(img, watermark)


def main():
    parser = argparse.ArgumentParser(description='Embed a fractal watermark into a 512x512 image crop')
    parser.add_argument('--image', help='input image; default = synthetic random 512')
    parser.add_argument('--out', default='results/watermarked_512.png')
    parser.add_argument('--box-out', default='results/watermark_box.json',
                        help='crop box record (decode.py re-crops with it)')
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

    img = Image.open(args.image).convert('RGB') if args.image else None
    if img is not None:
        box = center_box(img)
        crop = crop_to_tensor(img, box).to(pipeline.device)
    else:
        box, crop = None, torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=pipeline.device) * 2 - 1

    watermark = generate(pipeline, r=args.r, m=args.m, o=args.o,
                         x0=args.x0, a=args.a, k=args.k, d=args.d)
    img_wtm = embed(pipeline, crop, watermark)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if img is not None:
        paste_back(img, img_wtm, box).save(args.out)
        with open(args.box_out, 'w') as f:
            json.dump({'box': list(box), 'image': args.image}, f)
        print(f'crop (x, y, w, h): {box} -> pasted back at the same spot')
        print(f'box record -> {args.box_out}')
    else:
        to_pil(img_wtm).save(args.out)
    print(f'embed PSNR: {psnr(crop, img_wtm):.2f} dB')
    print(f'watermarked image -> {args.out}')


if __name__ == '__main__':
    main()
