"""
StarGAN v2 runner: reference-guided synthesis (source pose/expression,
target identity), CelebA-HQ checkpoint.

Executed by attacks/deepfakes.py with cwd=models/stargan-v2 and its venv.
"""

import os
import sys
from argparse import Namespace

sys.path.insert(0, os.getcwd())                        # stargan-v2 repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

import torch

from _common import load_rgb, output_path, parse_args, save_rgb, target_size

CONFIG = dict(img_size=256, style_dim=64, latent_dim=16, num_domains=2, w_hpf=1)
CHECKPOINT = 'expr/checkpoints/celeba_hq/100000_nets_ema.ckpt'
WING = 'expr/checkpoints/wing.ckpt'


def load_model():
    from core.model import build_model

    args = Namespace(**CONFIG, wing_path=WING)
    _, nets_ema = build_model(args)
    state = torch.load(CHECKPOINT, map_location='cpu')
    for name in ('generator', 'style_encoder', 'mapping_network'):
        missing, unexpected = nets_ema[name].module.load_state_dict(state[name], strict=False)
        # released checkpoint omits the constant high-pass kernel only
        assert not unexpected and set(missing) <= {'hpf.filter'}, (name, missing, unexpected)
    nets_ema = {name: net.eval().cuda() for name, net in nets_ema.items()}
    return args, nets_ema


def reenact(args, nets_ema, source, target):
    with torch.no_grad():
        masks = nets_ema['fan'].get_heatmap(target) if args.w_hpf > 0 else None
        style = nets_ema['style_encoder'](source, torch.zeros(1, dtype=torch.long, device=source.device))
        return nets_ema['generator'](target, style, masks=masks)


def main():
    args = parse_args('StarGAN v2')
    cfg, nets_ema = load_model()

    for source_path, target_path in args.pairs:
        source = load_rgb(source_path, (cfg.img_size, cfg.img_size)).cuda() * 2 - 1
        target = load_rgb(target_path, (cfg.img_size, cfg.img_size)).cuda() * 2 - 1
        fake = reenact(cfg, nets_ema, source, target)
        save_rgb((fake + 1) / 2, output_path(args.output_dir, target_path), size=target_size(target_path))


if __name__ == '__main__':
    main()
