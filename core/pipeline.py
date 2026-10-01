"""
FractalForensics core pipeline — paper §3.2-3.3.

One object wraps the whole system: fractal watermark generation, embedding
(encoder), recovery (decoder) and the training discriminator.

Tensor conventions (matching the pretrained checkpoints):
    images      float32 (B, 3, H, W) in [-1, 1]
    watermarks  float32 (B, 4, wtm_size, wtm_size) of 0/1, channel 0 = MSB
"""

import torch

from core.discriminator import Discriminator
from core.network import FractalForensics, WatermarkDecoder, load_encoder_decoder
from core.watermark import generate_watermark as _generate_watermark


class FractalForensicsPipeline:
    """Encoder, decoder and discriminator for one resolution / watermark size."""

    def __init__(self, img_size=256, wtm_size=8, latent_channels=64,
                 img_blocks=5, wtm_blocks=4, rec_blocks=4, dec_blocks=3,
                 disc_blocks=3, device='cpu'):
        self.img_size = img_size
        self.wtm_size = wtm_size
        self.latent_channels = latent_channels
        self.device = torch.device(device)

        self.encoder = FractalForensics(img_size, wtm_size, latent_channels,
                                        img_blocks, wtm_blocks, rec_blocks).to(self.device)
        self.decoder = WatermarkDecoder(latent_channels, dec_blocks).to(self.device)
        self.discriminator = Discriminator(latent_channels, disc_blocks).to(self.device)
        self.eval()

    @classmethod
    def from_weights(cls, weights_dir, img_size=256, wtm_size=8, latent_channels=64,
                     dec_blocks=3, disc_blocks=3, device='cpu'):
        """Build with pretrained encoder/decoder from a checkpoint directory.

        Block counts are inferred from the checkpoint, so any officially
        trained resolution loads without extra configuration. The discriminator
        has no released checkpoint and is randomly initialized (training only).
        """
        pipeline = cls(img_size=img_size, wtm_size=wtm_size,
                       latent_channels=latent_channels,
                       dec_blocks=dec_blocks, disc_blocks=disc_blocks, device=device)
        pipeline.encoder, pipeline.decoder, _ = load_encoder_decoder(
            weights_dir, img_size=img_size, wtm_size=wtm_size,
            latent_channels=latent_channels, device=device)
        return pipeline

    # ------------------------------------------------------------ watermark

    def generate_watermark(self, **params):
        """Paper §3.2: user parameters -> (1, 4, wtm_size, wtm_size) on device.

        See core.watermark.generate_watermark for the parameter ranges
        (r, m, o, x0, a, k, d, n).
        """
        watermark = _generate_watermark(**params)
        if watermark.shape[-1] != self.wtm_size:
            raise ValueError(
                f'Generated {watermark.shape[-1]}x{watermark.shape[-1]} watermark, '
                f'but this pipeline expects {self.wtm_size}x{self.wtm_size} (n={self.wtm_size.bit_length() - 1}).'
            )
        return watermark.to(self.device)

    # ------------------------------------------------------------- network

    def embed(self, img, watermark):
        """Embed the watermark: (B,3,H,W), (B,4,wtm,wtm) -> watermarked image."""
        self._check_image(img)
        return self.encoder(img, watermark)

    def decode(self, img):
        """Recover per-entry bit probabilities (B,4,wtm,wtm) from an image."""
        self._check_image(img)
        return self.decoder(img)

    def discriminate(self, img):
        """Discriminator logits (B,1), higher = judged watermarked (training only)."""
        self._check_image(img)
        return self.discriminator(img)

    # ---------------------------------------------------------------- modes

    def eval(self):
        """Switch all submodules to evaluation mode (dropout/batchnorm off)."""
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        return self

    def train(self):
        """Switch all submodules to training mode."""
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        return self

    def _check_image(self, img):
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f'Expected image (B, 3, H, W), got {tuple(img.shape)}.')
        if img.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f'Expected {self.img_size}x{self.img_size} images, got {tuple(img.shape[-2:])}.'
            )
