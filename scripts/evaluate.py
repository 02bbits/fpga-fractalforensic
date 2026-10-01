"""
FractalForensics evaluation — embed, attack, recover, score, localize.

Run from the project root:
    python3 scripts/evaluate.py                     # synthetic image, pretrained weights
    python3 scripts/evaluate.py --image photo.jpg   # real image
    python3 scripts/evaluate.py --weights path/to/weights

Outputs:
    - bit/patch recovery rates per attack (paper Table 2)
    - results/localization.png: red overlay on entries lost after a simulated edit
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from attacks.manipulations import ATTACKS
from core import FractalForensicsPipeline
from core.metrics import bit_accuracy, patch_accuracy, patch_error_map

IMG_SIZE = 256
WTM_SIZE = 8
PATCH = IMG_SIZE // WTM_SIZE


def load_image(path):
    """PIL image -> (1, 3, 256, 256) tensor in [-1, 1]; synthetic if path is None."""
    if path is None:
        return torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 2 - 1
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return (TF.to_tensor(img) * 2 - 1).unsqueeze(0)


def save_image(tensor, path):
    """Save an RGB tensor (3, H, W) already in [0, 1] (e.g. the localization overlay)."""
    TF.to_pil_image(tensor.clamp(0, 1)).save(path)


def tamper(imgs_wtm, imgs_clean, x0=64, y0=64, w=128, h=128):
    """Simulated edit (paper §4.4 cropping experiment): restore original pixels
    in a region of the watermarked image, wiping the embedded entries there."""
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
    parser = argparse.ArgumentParser(description='FractalForensics evaluation')
    parser.add_argument('--image', help='input image path; default = synthetic random image')
    parser.add_argument('--weights', default='FractalForensics/weights/256',
                        help='dir containing encoder.pth / decoder.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    pipeline = FractalForensicsPipeline.from_weights(
        args.weights, img_size=IMG_SIZE, wtm_size=WTM_SIZE, device=args.device)
    watermark = pipeline.generate_watermark()
    img = load_image(args.image).to(pipeline.device)

    with torch.no_grad():
        img_wtm = pipeline.embed(img, watermark)

    print(f'\n{"attack":<16}{"bit acc":>10}{"patch acc":>12}')
    print('-' * 38)
    for name, attack in ATTACKS.items():
        with torch.no_grad():
            watermark_rec = pipeline.decode(attack(img_wtm))
        print(f'{name:<16}{bit_accuracy(watermark, watermark_rec):>10.4f}'
              f'{patch_accuracy(watermark, watermark_rec):>12.4f}')

    img_tampered = tamper(img_wtm, img)
    with torch.no_grad():
        watermark_rec = pipeline.decode(img_tampered)
    print(f'\ntamper region 128x128 px (4x4 of the {WTM_SIZE}x{WTM_SIZE} entries): '
          f'patch acc {patch_accuracy(watermark, watermark_rec):.4f}')

    overlay = localization_overlay(img_tampered, patch_error_map(watermark, watermark_rec))
    os.makedirs('results', exist_ok=True)
    save_image(overlay, 'results/localization.png')
    print('localization overlay -> results/localization.png')


if __name__ == '__main__':
    main()
