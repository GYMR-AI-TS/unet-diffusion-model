### encoder.py
### Implements the UNet Encoder


import torch.nn as nn


class EncoderSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 2),
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
        x1 = self.block1(input)  # (64, 64, 30, 30)
        x = self.maxpool(x1)  # (64, 64, 15, 15)
        x2 = self.block2(x)  # (64, 128, 13, 13)
        x = self.maxpool(x2)  # (64, 128, 6, 6)
        skip_connections = [x1, x2]
        return x, skip_connections
