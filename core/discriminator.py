"""
FractalForensics discriminator — paper §3.3, used only for adversarial training
(L_adv / L_D in §3.4). Not needed for watermark embedding, recovery or scoring.
"""

import torch.nn as nn

from core.blocks import ConvBlock, SEResBlock


class Discriminator(nn.Module):
    """Binary classifier: watermarked vs original image."""

    def __init__(self, latent_channels=64, num_blocks=3):
        super().__init__()
        self.conv_head = ConvBlock(3, latent_channels, stride=2)
        self.resblocks = nn.Sequential(*[SEResBlock(latent_channels) for _ in range(num_blocks)])
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(latent_channels, 1),
        )

    def forward(self, x):
        return self.fc(self.resblocks(self.conv_head(x)))
