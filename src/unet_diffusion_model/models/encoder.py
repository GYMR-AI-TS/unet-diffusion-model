### encoder.py
###


import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.activation = nn.SiLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.activation(x)
        x = self.conv2(x)
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = EncoderBlock(3, 64)
        self.block2 = EncoderBlock(64, 128)
        self.block3 = EncoderBlock(128, 256)
        self.block4 = EncoderBlock(256, 512)

    def forward(self, input):
        x1 = self.block1(input)
        x = self.maxpool(x1)
        x2 = self.block2(x)
        x = self.maxpool(x2)
        x3 = self.block3(x)
        x = self.maxpool(x3)
        x4 = self.block4(x)
        x = self.maxpool(x4)
        skip_connections = [x1, x2, x3, x4]
        return x, skip_connections
