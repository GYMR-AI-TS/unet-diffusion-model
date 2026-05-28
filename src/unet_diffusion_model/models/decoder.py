### decoder.py
### Implements the UNet Decoder

import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop


class DecoderSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 2),
            nn.SiLU()
        )

    def forward(self, input):
        return self.block(input)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block3 = nn.Sequential(
            DecoderSubBlock(256, 128),
            nn.ConvTranspose2d(128, 64, 3, 2)
        )
        self.block4 = DecoderSubBlock(128, 64)

    def forward(self, input, skip_connections):
        # (64, 128, 9, 9), (64, 128, 13, 13)
        x = torch.cat([center_crop(skip_connections[1], 9), input], dim=1)  # (64, 256, 9, 9)
        x = self.block3(x)  # (64, 128, 15, 15) + (64, 64, 30, 30)
        x = torch.cat([center_crop(skip_connections[0], 15),  x], dim=1)  # (64, 128, 15, 15)
        return self.block4(x)  # (64, 64, 13, 13)
