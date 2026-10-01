"""
Benign image-processing attacks on watermarked images (paper Table 2).

Pure torch/PIL — no kornia/lpips needed. Attack names match the paper and the
vendored repo's configuration strings, selected via get_attack() instead of eval.
"""

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


class Identity:
    """No-op (baseline: untouched watermarked image)."""

    def __call__(self, img):
        return img


class Resize:
    """Downscale then restore to original size (nearest)."""

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img):
        down = F.interpolate(img, scale_factor=self.ratio, mode='nearest')
        return F.interpolate(down, size=img.shape[-2:], mode='nearest')


class GaussianNoise:
    """Additive white Gaussian noise, std 0.1 (hardest benign op in the paper)."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        return img + torch.randn_like(img) * self.std


class GaussianBlur:
    """Gaussian blur, kx1 kernel with sigma (via torchvision, same math as kornia)."""

    def __init__(self, sigma=2.0, kernel=3):
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self, img):
        return TF.gaussian_blur(img, self.kernel, self.sigma)


class MedianBlur:
    """Median blur via sliding-window median (torch.unfold), kernel 3."""

    def __init__(self, kernel=3):
        self.kernel = kernel

    def __call__(self, img):
        k = self.kernel
        p = k // 2
        padded = F.pad(img, (p, p, p, p), mode='replicate')
        win = padded.unfold(2, k, 1).unfold(3, k, 1)          # (B,C,H,W,k,k)
        return win.contiguous().view(*img.shape, k * k).median(dim=-1).values


class Jpeg:
    """JPEG compression via PIL round-trip at the given quality (in-memory)."""

    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img):
        out = torch.empty_like(img)
        for i in range(img.shape[0]):
            x = img[i].clamp(-1, 1).permute(1, 2, 0).detach().cpu().numpy()
            x = ((x + 1) / 2 * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(x).save(buf, format='JPEG', quality=self.quality)
            y = np.array(Image.open(buf), dtype=np.float32) / 255 * 2 - 1
            out[i] = torch.from_numpy(y).permute(2, 0, 1).to(img.device)
        return out


ATTACKS = {
    'Identity()': Identity(),
    'Resize(0.5)': Resize(0.5),
    'GaussianNoise()': GaussianNoise(0.1),
    'GaussianBlur(2,3)': GaussianBlur(2.0, 3),
    'MedBlur(3)': MedianBlur(3),
    'Jpeg(50)': Jpeg(50),
}


def get_attack(name):
    return ATTACKS[name]
