"""
Watermark generation self-check (paper §3.2).

Run from the project root:
    python3 scripts/watermark_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # direct script execution

import numpy as np

from core.watermark import generate_watermark


def main():
    watermark = generate_watermark(r=1, m=2, o=3, x0=0.5, a=3.9, k=100, d=3, n=4)
    assert watermark.shape == (1, 4, 16, 16), watermark.shape
    assert set(watermark.unique().tolist()) <= {0.0, 1.0}
    print('generate_watermark OK:', tuple(watermark.shape), watermark.dtype)

    other = generate_watermark(r=1, m=2, o=3, x0=0.5, a=3.9, k=100, d=4, n=4)
    assert not np.allclose(watermark.numpy(), other.numpy())
    print('parameter change OK: encryption key d alters the watermark')


if __name__ == '__main__':
    main()
