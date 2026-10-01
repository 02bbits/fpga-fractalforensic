"""
Watermark scoring — paper §3.4 / §4: bit-wise and patch-wise recovery rates,
plus the per-patch error map used for localization overlays.
"""

import torch


def bit_accuracy(w_gt, w_rec):
    """Fraction of correctly recovered bits over the batch, shape (B,4,H,W)."""
    return ((w_gt >= 0.5) == (w_rec >= 0.5)).float().mean().item()


def patch_accuracy(w_gt, w_rec):
    """Official metric: fraction of entries whose 4 bits ALL match, (B,H,W)."""
    return ((w_gt >= 0.5) == (w_rec >= 0.5)).all(dim=1).float().mean().item()


def patch_error_map(w_gt, w_rec):
    """(B,H,W) boolean map: True where a watermark entry was lost."""
    return ~((w_gt >= 0.5) == (w_rec >= 0.5)).all(dim=1)


def patch_accuracy_per_image(w_gt, w_rec):
    """Per-image patch accuracy, shape (B,) — the per-sample score used for AUC."""
    return ((w_gt >= 0.5) == (w_rec >= 0.5)).all(dim=1).float().mean(dim=(1, 2)).cpu().numpy()
