### encoder.py
### Implements the UNet Encoder


import torch.nn as nn


class EncoderSubBlock(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = EncoderSubBlock(3, 64)
        self.block2 = EncoderSubBlock(64, 128)

    def forward(self, input):
        # input : (batch_size, channels, height, width) = (64, 3, 32, 32)
        x1 = self.block1(input)  # (64, 64, 32, 32)
        x = self.maxpool(x1)  # (64, 64, 16, 16)
        x2 = self.block2(x)  # (64, 128, 16, 16)
        x = self.maxpool(x2)  # (64, 128, 8, 8)
        skip_connections = [x1, x2]
        return x, skip_connections
