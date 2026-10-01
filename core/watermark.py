"""
Watermark generation helpers.

Pipeline (see scripts/watermark_demo.py and generate_watermark below):
    1. hilbert_curve(n)          -> base path over 2^n x 2^n grid, len 2**(2n)
    2. apply_rotation / apply_mirroring / apply_order_modification
       -> transform the path (positions stay on the grid, visit ORDER changes)
    3. logistic-map chaotic digits fill the path -> W matrix
    4. to_4bit_binary_tensor(W)  -> [4, 2^n, 2^n] bit-plane tensor

Conventions: curve is a list of (x, y) tuples, x = column, y = row,
origin top-left. Route transforms were merged in from the old root-level
create_variant.py (rotate_route/mirror_route/reverse_route/shift_start);
hilbert_curve from test_hilbert.py.
"""

import numpy as np
import torch

# ======================= Helpers

def _transpose(curve):
    return [(y, x) for x, y in curve]

def extract_dth_digit(x, d):
    """
    d-th decimal digit (d=1 -> first after the point) of x in (0,1).
    """
    return int(x * 10 ** d) % 10

def to_4bit_binary_tensor(W):
    """[H, W] digit matrix (values 0..9) -> [4, H, W] uint8 bit-plane tensor.

    Channel 0 = LSB, channel 3 = MSB.
    """
    W = np.asarray(W, dtype=np.uint8)
    return ((W[None, :, :] >> np.arange(4)[:, None, None]) & 1).astype(np.uint8)


# ======================== Watermark Generation Pipelines
def hilbert_curve(n) -> list[tuple[int, int]]:
    """
    Hilbert path over the full 2^n x 2^n grid.

    Returns list of (x, y), length 2**(2n), in visit order.
    Lifted from test_hilbert.py (d2xy), bounds check removed.
    """
    route = []
    for d in range(4 ** n):
        x = y = 0
        s = 1
        t = d
        while s < 2 ** n:
            rx = 1 if (t & 2) else 0
            ry = 1 if (t & 1) ^ rx else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        route.append((x, y))
    return route


def apply_rotation(curve, r):
    """Rotate path r times by 90 deg (r in 0..3), around the grid's top-left
    (max_x - x flip on y): same as the old create_variant.rotate_route."""
    r %= 4
    for _ in range(r):
        max_x = max(x for x, y in curve)
        curve = [(y, max_x - x) for x, y in curve]
    return curve


def mirror_route(curve, axis="x"):
    """Flip about x-axis (max_x - x) or y-axis (max_y - y); merged from create_variant."""
    max_x = max(x for x, y in curve)
    max_y = max(y for x, y in curve)
    if axis == "x":
        return [(max_x - x, y) for x, y in curve]
    return [(x, max_y - y) for x, y in curve]


def shift_start(curve, index):
    """Rotate visit order so point `index` comes first; merged from create_variant."""
    return curve[index:] + curve[:index]


def apply_mirroring(curve, m):
    """
    D4 group member m % 8 (m in 0..8 per spec, % 8 keeps it total).

    0 identity | 1 flip-x | 2 flip-y | 3 flip-both
    4 transpose | 5 transpose+flip-x | 6 transpose+flip-y | 7 transpose+flip-both
    """
    m %= 8
    if m == 0:
        return curve
    if m == 4:
        return _transpose(curve)
    if m >= 5:  # diagonal mirrors = transpose then a axis mirror
        return apply_mirroring(_transpose(curve), m - 4)
    max_x = max(x for x, y in curve)
    max_y = max(y for x, y in curve)
    if m == 1:
        return [(max_x - x, y) for x, y in curve]
    if m == 2:
        return [(x, max_y - y) for x, y in curve]
    return [(max_x - x, max_y - y) for x, y in curve]


def apply_order_modification(curve, o):
    """
    Reorder the visit sequence (positions unchanged).

    0 none | 1 reverse | 2 zigzag (reverse subsequence on odd grid rows)
    | 3 cross-flip (swap x/y of each point)
    """
    if o == 1:
        return curve[::-1]
    if o == 2:
        out = list(curve)
        i = 0
        while i < len(out):
            j = i
            while j < len(out) and out[j][1] == out[i][1]:
                j += 1
            if out[i][1] % 2 == 1:
                out[i:j] = out[i:j][::-1]
            i = j
        return out
    if o == 3:
        return _transpose(curve)
    return curve


# ======================== Full Generation Pipeline

def generate_watermark(r=1, m=2, o=3, x0=0.5, a=3.9, k=100, d=3, n=3):
    """Paper §3.2: user parameters -> encrypted fractal watermark.

    r/m/o vary the curve, x0/a/k/d drive the logistic-map one-way encryption.
    Returns a float32 tensor (1, 4, 2^n, 2^n) of 0/1 bits in the network
    convention: channel 0 = MSB (bit 3). `to_4bit_binary_tensor` is LSB-first,
    so the channels are flipped to match the pretrained encoder/decoder.
    """
    curve = hilbert_curve(n)
    curve = apply_rotation(curve, r)
    curve = apply_mirroring(curve, m)
    curve = apply_order_modification(curve, o)

    x = x0
    for _ in range(k):
        x = a * x * (1 - x)
    digits = []
    for _ in range(2 ** (2 * n)):
        x = a * x * (1 - x)
        digits.append(extract_dth_digit(x, d))

    w = np.zeros((2 ** n, 2 ** n), dtype=np.uint8)
    for idx, (px, py) in enumerate(curve):
        w[px, py] = digits[idx]

    wc = to_4bit_binary_tensor(w)              # (4, H, W), LSB-first
    return torch.from_numpy(wc).float().flip(0).unsqueeze(0)
