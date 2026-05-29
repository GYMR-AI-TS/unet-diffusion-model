### decoder.py
### Implements the UNet Decoder

import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop


class DecoderSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, input):
        return self.block(input)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block3 = nn.Sequential(
            DecoderSubBlock(256, 128), nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.block4 = DecoderSubBlock(128, 64)

    def forward(self, input, skip_connections):
        # (64, 128, 16, 16) + (64, 128, 16, 16)
        x = torch.cat(
            [center_crop(skip_connections[1], 16), input], dim=1
        )  # (64, 256, 16, 16)
        x = self.block3(x)  # (64, 64, 32, 32) + (64, 64, 32, 32)
        x = torch.cat(
            [center_crop(skip_connections[0], 32), x], dim=1
        )  # (64, 128, 32, 32)
        return self.block4(x)  # (64, 64, 32, 32)
