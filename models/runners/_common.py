"""Shared helpers for the model runners in this directory.

Runners are executed by attacks/deepfakes.py with the model's own venv python
and with cwd set to the model repo, so repo-relative paths resolve.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pairs', nargs='+', required=True, metavar='SOURCE,TARGET',
                        help='one "source_path,target_path" entry per image pair')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    args.pairs = [tuple(pair.split(',')) for pair in args.pairs]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def target_size(path):
    """(width, height) of an image on disk."""
    with Image.open(path) as img:
        return img.size


def load_rgb(path, size=None):
    """Image file -> float tensor (1, 3, H, W) in [0, 1]."""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return x.unsqueeze(0)


def save_rgb(tensor, path, size=None):
    """(1, 3, H, W) or (3, H, W) tensor in [0, 1] -> image file."""
    x = tensor.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if size is not None:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(size[1], size[0]), mode='bilinear', align_corners=False)[0]
    x = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def output_path(output_dir, target_path):
    return str(Path(output_dir) / (Path(target_path).stem + '.png'))
