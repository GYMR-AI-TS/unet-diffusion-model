### decoder.py
### Implements the UNet Decoder

import torch
import torch.nn as nn


class DecoderSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.SiLU()
        )

    def forward(self, input):
        return self.block(input)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampling
        self.block1 = nn.Sequential(
            DecoderSubBlock(512, 256),
            nn.ConvTranspose2d(256, 256, 2)
        )
        self.block2 = nn.Sequential(
            DecoderSubBlock(256, 128),
            nn.ConvTranspose2d(128, 128, 2)
        )
        self.block3 = nn.Sequential(
            DecoderSubBlock(128, 64),
            nn.ConvTranspose2d(64, 64, 2)
        )
        self.block4 = DecoderSubBlock(64, 3)

    def forward(self, input, skip_connections):
        x = torch.cat([skip_connections[0], input], dim=1)
        x = self.block1(x)
        x = torch.cat([skip_connections[1], x], dim=1)
        x = self.block2(x)
        x = torch.cat([skip_connections[2], x], dim=1)
        x = self.block3(x)
        x = torch.cat([skip_connections[3], x], dim=1)
        return self.block4(x)
