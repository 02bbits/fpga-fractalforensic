"""
Training losses — paper §3.4. The repo ships no training loop; these are the
components needed when one is built (encoder, decoder and discriminator all
live in core/). LPIPS is optional and only imported on demand.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Paper §4.1 loss weights: (MSE, LPIPS, adversarial, decoder).
DEFAULT_LOSS_WEIGHTS = {'mse': 1.0, 'lpips': 0.5, 'adv': 0.01, 'dec': 12.0}


def mse_loss(img, img_rec):
    """L_MSE = ||I - I_rec||² (eq. 2)."""
    return F.mse_loss(img_rec, img)


def decoder_loss(wtm_rec, wtm_gt):
    """L_dec = ||w_rec - w|| (eq. 6), L1 on the recovered 4-bit entries."""
    return F.l1_loss(wtm_rec, wtm_gt)


def lpips_loss(img, img_rec, metric):
    """L_LPIPS = perceptual distance (eq. 3). `metric` comes from build_lpips()."""
    return metric(img_rec, img).mean()


def discriminator_loss(d_real_logits, d_fake_logits):
    """L_D = -E log D(I) + E log(1 - D(I_rec)) (eq. 4)."""
    real = F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits))
    fake = F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
    return real + fake


def adversarial_loss(d_fake_logits):
    """L_adv = -E log D(I_rec) (eq. 5), the encoder's adversarial objective."""
    return F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))


def total_loss(img, img_rec, wtm_rec, wtm_gt, d_fake_logits=None,
               lpips_metric=None, weights=None):
    """Weighted sum of the objectives (eq. 7).

    L_adv is only included when d_fake_logits and an lpips_metric-for-LPIPS are
    provided; terms with weight 0 are skipped.
    """
    weights = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}
    loss = weights['mse'] * mse_loss(img, img_rec)
    loss = loss + weights['dec'] * decoder_loss(wtm_rec, wtm_gt)
    if weights['lpips'] > 0:
        if lpips_metric is None:
            raise ValueError('An LPIPS metric is required when its loss weight is non-zero.')
        loss = loss + weights['lpips'] * lpips_loss(img, img_rec, lpips_metric)
    if d_fake_logits is not None and weights['adv'] > 0:
        loss = loss + weights['adv'] * adversarial_loss(d_fake_logits)
    return loss


def build_lpips(device='cpu'):
    """Load the pretrained AlexNet LPIPS metric (paper §4.1) on demand."""
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            'LPIPS loss needs the `lpips` package: pip install lpips'
        ) from exc
    metric = lpips.LPIPS(net='alex', pretrained=True).to(device)
    metric.eval()
    return metric
