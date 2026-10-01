"""
FractalForensics watermarking network — paper §3.3, Figure 2 (encoder + decoder).

Parameter names mirror the official implementation
(FractalForensics/model/network.py) so released checkpoints load unchanged.

    FractalForensics (encoder):  I, Wc -> I_rec    watermarked image
        ImageFeatureExtractor    7x7 conv + N_img SEResBlocks (spatial preserved)
        WatermarkDiffusion       entry-to-patch nearest upsample + refinement
        fusion + recon           concat features, N_rec SEResBlocks, skip to I
    WatermarkDecoder (decoder):  I_att -> w_rec    per-entry bit probabilities
"""

import os
import re

import torch
import torch.nn as nn

from core.blocks import ConvBlock, SEResBlock


class ImageFeatureExtractor(nn.Module):
    """Image features in the pixel-aligned domain (spatial size unchanged)."""

    def __init__(self, out_channels=64, num_blocks=5):
        super().__init__()
        self.conv_head = ConvBlock(3, out_channels, kernel_size=7, stride=1, padding=3)
        self.resblocks = nn.Sequential(*[SEResBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.resblocks(self.conv_head(x))


class WatermarkDiffusion(nn.Module):
    """Entry-to-patch: each 4-bit entry is expanded to its image patch."""

    def __init__(self, img_size, wtm_size, out_channels=64, num_blocks=4):
        super().__init__()
        self.img_size = img_size
        self.wtm_size = wtm_size
        self.patch_size = img_size // wtm_size
        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        self.watermark_feature_extractor = nn.Sequential(
            ConvBlock(4, 32),
            *[SEResBlock(32) for _ in range(num_blocks)],
            ConvBlock(32, out_channels),
        )

    def forward(self, watermark):
        return self.watermark_feature_extractor(self.upsample(watermark))


class FractalForensics(nn.Module):
    """Encoder: invisibly embed the fractal watermark into the image."""

    def __init__(self, img_size=256, wtm_size=8, latent_channels=64,
                 img_blocks=5, wtm_blocks=4, rec_blocks=4):
        super().__init__()
        self.image_extractor = ImageFeatureExtractor(out_channels=latent_channels,
                                                     num_blocks=img_blocks)
        self.watermark_diffuser = WatermarkDiffusion(img_size=img_size,
                                                     wtm_size=wtm_size,
                                                     out_channels=latent_channels,
                                                     num_blocks=wtm_blocks)

        intermediate_channels = latent_channels * 3 // 2
        self.fusion = nn.Sequential(
            ConvBlock(latent_channels * 2, intermediate_channels),
            ConvBlock(intermediate_channels, latent_channels),
        )
        self.resblocks = nn.Sequential(*[SEResBlock(latent_channels) for _ in range(rec_blocks)])
        self.recon = nn.Sequential(
            ConvBlock(latent_channels + 3, 32),
            ConvBlock(32, 3, activation=False),
        )

    def forward(self, img, wtm):
        img_features = self.image_extractor(img)
        wtm_features = self.watermark_diffuser(wtm)
        fused = self.fusion(torch.cat([img_features, wtm_features], dim=1))
        fused = self.resblocks(fused)
        fused_cat = torch.cat([fused, img], dim=1)
        return torch.clamp(self.recon(fused_cat), min=-1, max=1)


class WatermarkDecoder(nn.Module):
    """Recover the 4-bit watermark matrix from a (possibly attacked) image."""

    def __init__(self, latent_channels=64, num_blocks=3):
        super().__init__()
        self.conv_head = nn.Sequential(
            ConvBlock(3, latent_channels // 2, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.2),
            ConvBlock(latent_channels // 2, latent_channels, stride=2),
            nn.Dropout2d(p=0.2),
            ConvBlock(latent_channels, latent_channels, stride=2),
        )
        self.resblocks = nn.Sequential(*[SEResBlock(latent_channels) for _ in range(num_blocks)])
        self.conv_tail = nn.Sequential(
            ConvBlock(latent_channels, latent_channels, stride=2),
            nn.Dropout2d(p=0.1),
            ConvBlock(latent_channels, latent_channels // 2, stride=2),
            nn.Dropout2d(p=0.05),
            ConvBlock(latent_channels // 2, latent_channels // 2, stride=2),
            nn.Conv2d(latent_channels // 2, 4, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_head(x)
        x = self.resblocks(x)
        return self.conv_tail(x)


# ======================== Checkpoint Utilities

def _num_modules(state_dict, prefix):
    """Number of modules in a Sequential, from key indices like ``prefix.3``."""
    return len({m.group(1) for key in state_dict
                for m in [re.match(rf'^{prefix}\.(\d+)\.', key)] if m})


def infer_block_counts(encoder_state, decoder_state):
    """Block counts of an officially trained checkpoint.

    The released 256px checkpoint uses more watermark-diffusion blocks than
    the paper configs suggest, so counts are read from the weights themselves.
    """
    return {
        'img_blocks': _num_modules(encoder_state, 'image_extractor.resblocks'),
        'wtm_blocks': _num_modules(encoder_state, 'watermark_diffuser.watermark_feature_extractor') - 2,
        'rec_blocks': _num_modules(encoder_state, 'resblocks'),
        'dec_blocks': _num_modules(decoder_state, 'resblocks'),
    }


def load_encoder_decoder(weights_dir, img_size=256, wtm_size=8, latent_channels=64, device='cpu'):
    """Build the encoder and decoder and load ``encoder.pth`` / ``decoder.pth``.

    Block counts are inferred from the checkpoint; img_size, wtm_size and
    latent_channels are architecture parameters not stored in the weights.
    """
    encoder_path = os.path.join(weights_dir, 'encoder.pth')
    decoder_path = os.path.join(weights_dir, 'decoder.pth')
    if not (os.path.isfile(encoder_path) and os.path.isfile(decoder_path)):
        raise FileNotFoundError(
            f'No encoder.pth / decoder.pth found in {weights_dir!r}.'
        )

    encoder_state = torch.load(encoder_path, map_location=device)
    decoder_state = torch.load(decoder_path, map_location=device)
    blocks = infer_block_counts(encoder_state, decoder_state)

    encoder = FractalForensics(img_size, wtm_size, latent_channels,
                               blocks['img_blocks'], blocks['wtm_blocks'],
                               blocks['rec_blocks']).to(device)
    decoder = WatermarkDecoder(latent_channels, blocks['dec_blocks']).to(device)
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, blocks
