"""
Golden-reference dump of one ConvBlock / SEResBlock for hardware testbenches:
weights (raw + BatchNorm folded into the conv), input and output.

Run from the project root:
    python3 scripts/dump_block.py --block conv --in 3 --out 8 --hw 16 16
    python3 scripts/dump_block.py --block seres --in 64 --hw 8 8 \
        --weights weights/256/encoder.pth --prefix image_extractor.resblocks.0.
    python3 scripts/dump_block.py --block conv --in 1 --out 1 --state my.pt --input x.npy

Outputs (in --outdir, default results/block_dump/):
    input.npy, output.npy, <param>.npy for every state_dict entry,
    folded_conv_weight.npy / folded_conv_bias.npy  (conv' = BN(conv), what hardware runs)
All float32, row-major (C order).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution

import numpy as np
import torch
import torch.nn.functional as F

from core.blocks import ConvBlock, SEResBlock


def fold_bn(conv_block):
    """ConvBlock (conv, no bias) + eval BN -> equivalent conv weight and bias."""
    bn = conv_block.bn
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv_block.conv.weight * scale.view(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    return weight, bias


def main():
    parser = argparse.ArgumentParser(description='Dump a block\'s weights, input and output')
    parser.add_argument('--block', choices=['conv', 'seres'], default='conv')
    parser.add_argument('--in', dest='in_ch', type=int, default=3, help='input channels')
    parser.add_argument('--out', dest='out_ch', type=int, help='output channels (conv only; seres = --in)')
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, help='default kernel // 2')
    parser.add_argument('--no-act', action='store_true', help='ConvBlock without LeakyReLU')
    parser.add_argument('--hw', type=int, nargs=2, default=[8, 8], metavar=('H', 'W'))
    parser.add_argument('--weights', help='checkpoint .pth to take this block from (needs --prefix)')
    parser.add_argument('--prefix', default='', help="e.g. 'image_extractor.resblocks.0.'")
    parser.add_argument('--state', help='custom state_dict (.pt) with the keys this block prints')
    parser.add_argument('--input', help='custom input .npy of shape (1, in, H, W) or (in, H, W)')
    parser.add_argument('--seed', type=int, default=0, help='random weights/input when not given')
    parser.add_argument('--outdir', default='results/block_dump')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    padding = args.kernel // 2 if args.padding is None else args.padding
    if args.block == 'conv':
        block = ConvBlock(args.in_ch, args.out_ch or args.in_ch, args.kernel, args.stride,
                          padding, activation=not args.no_act)
        conv_block = block
    else:
        block = SEResBlock(args.in_ch)
        conv_block = block.conv

    if args.weights:
        sd = torch.load(args.weights, map_location='cpu')
        block.load_state_dict({k.removeprefix(args.prefix): v for k, v in sd.items()
                               if k.startswith(args.prefix)})
    elif args.state:
        block.load_state_dict(torch.load(args.state, map_location='cpu'))
    else:  # random but non-trivial BN so folding is actually exercised
        with torch.no_grad():
            bn = conv_block.bn
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.5, 0.5)
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
    block.eval()

    if args.input:
        x = torch.from_numpy(np.load(args.input)).float()
        x = x[None] if x.dim() == 3 else x
    else:
        x = torch.randn(1, args.in_ch, *args.hw)

    with torch.no_grad():
        y = block(x.clone())  # clone: blocks use inplace LeakyReLU
        weight, bias = fold_bn(conv_block)
        # self-check: folded conv reproduces the block's conv stage
        ref = F.conv2d(x, weight, bias, conv_block.conv.stride, conv_block.conv.padding)
        ref = conv_block.activation(ref)
        assert torch.allclose(ref, conv_block(x.clone()), atol=1e-5), 'BN folding mismatch'

    arrays = {'input': x, 'output': y, 'folded_conv_weight': weight, 'folded_conv_bias': bias,
              **{k: v for k, v in block.state_dict().items() if v.dim() > 0}}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in arrays.items():
        a = tensor.detach().numpy().astype(np.float32)
        np.save(out / f'{name}.npy', a)
        print(f'{name:28s} {str(tuple(a.shape)):18s} min {a.min():+.4f} max {a.max():+.4f}')
    print(f'BN folding self-check OK -> {out}/')


if __name__ == '__main__':
    main()
