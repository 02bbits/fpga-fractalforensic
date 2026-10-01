"""
UniFace runner: face swapping at 512px using the released swap checkpoint.

UNVERIFIED locally: needs the repo's custom CUDA ops (training/op) compiled
and more VRAM than the local 4GB. Executed by attacks/deepfakes.py with
cwd=models/UniFace; see models/README.md for setup notes.
"""

import os
import sys

sys.path.insert(0, os.getcwd())                        # UniFace repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

import torch

from _common import load_rgb, output_path, parse_args, save_rgb, target_size

CHECKPOINT = 'session/swap/checkpoints/500000.pt'


def load_model():
    import generate_swap

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    # generate_swap.Model reads a module-level `args` built only when the
    # script runs as __main__; inject the training config captured in the ckpt.
    generate_swap.args = state['train_args']
    model = generate_swap.Model('cuda').half().cuda()
    model.g_ema.load_state_dict(state['g_ema'])
    model.e_ema.load_state_dict(state['e_ema'])
    return model.eval(), state['train_args'].size


def main():
    args = parse_args('UniFace')
    model, size = load_model()

    for source_path, target_path in args.pairs:
        source = load_rgb(source_path, (size, size)).cuda().half()
        target = load_rgb(target_path, (size, size)).cuda().half()
        with torch.no_grad():
            _, _, fake = model([target, source])
        save_rgb(fake.float(), output_path(args.output_dir, target_path),
                 size=target_size(target_path))


if __name__ == '__main__':
    main()
