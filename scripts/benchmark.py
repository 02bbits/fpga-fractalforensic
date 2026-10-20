"""
Per-stage latency: create watermark (CPU, §3.2), embed (encoder), decode
(decoder), discriminate (discriminator) — the software baseline for the FPGA.

Run from the project root:
    python3 scripts/benchmark.py                          # 256px, cuda if available
    python3 scripts/benchmark.py --size 512 --device cpu --iters 20
    python3 scripts/benchmark.py --batch 4                # per-call and per-image times

Network stages are timed with CUDA synchronization so GPU numbers are real
latency, not kernel-launch time. The discriminator has no released weights;
random init has identical cost.
"""

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling scripts

import torch

from core import FractalForensicsPipeline
from eval_celeba import CONFIGS, random_watermark_params


def timed(fn, device, iters, warmup):
    """Run fn warmup+iters times; return per-call wall times in ms (post-warmup)."""
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    times = []
    for i in range(warmup + iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    return times


def main():
    parser = argparse.ArgumentParser(description='FractalForensics per-stage timing')
    parser.add_argument('--size', type=int, default=256, choices=sorted(CONFIGS))
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = CONFIGS[args.size]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pipeline = FractalForensicsPipeline.from_weights(
        f'weights/{args.size}', img_size=args.size, wtm_size=cfg['wtm_size'],
        latent_channels=cfg['latent_channels'], device=args.device)

    img = torch.rand(args.batch, 3, args.size, args.size, device=device) * 2 - 1
    wtm = torch.cat([pipeline.generate_watermark(n=cfg['n'], **random_watermark_params(rng))
                     for _ in range(args.batch)])
    with torch.inference_mode():
        img_wtm = pipeline.embed(img, wtm)

    def create():
        for _ in range(args.batch):
            pipeline.generate_watermark(n=cfg['n'], **random_watermark_params(rng))

    stages = {
        'create watermark': create,
        'embed watermark': lambda: pipeline.embed(img, wtm),
        'decode watermark': lambda: pipeline.decode(img_wtm),
        'discriminate': lambda: pipeline.discriminate(img_wtm),
    }
    params = {
        'embed watermark': pipeline.encoder,
        'decode watermark': pipeline.decoder,
        'discriminate': pipeline.discriminator,
    }

    gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    print(f'{args.size}px, wtm {cfg["wtm_size"]}x{cfg["wtm_size"]}, batch {args.batch}, '
          f'{args.iters} iters ({args.warmup} warmup), device {gpu}\n')
    print(f'{"stage":<18}{"params":>10}{"median ms":>12}{"mean ms":>10}{"std":>8}'
          f'{"min ms":>9}{"ms/img":>9}')
    print('-' * 76)
    total = 0.0
    with torch.inference_mode():
        for name, fn in stages.items():
            t = timed(fn, device, args.iters, args.warmup)
            med = statistics.median(t)
            total += med
            n_params = sum(p.numel() for p in params[name].parameters()) if name in params else 0
            print(f'{name:<18}{n_params / 1e6:>9.2f}M{med:>12.3f}{statistics.mean(t):>10.3f}'
                  f'{statistics.stdev(t) if len(t) > 1 else 0:>8.3f}{min(t):>9.3f}'
                  f'{med / args.batch:>9.3f}')
    print('-' * 76)
    print(f'{"total (medians)":<18}{"":>10}{total:>12.3f}{"":>27}{total / args.batch:>9.3f}')


if __name__ == '__main__':
    main()
