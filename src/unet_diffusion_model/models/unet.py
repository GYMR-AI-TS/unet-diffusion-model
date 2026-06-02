### unet.py
### Implements UNet

import torch.nn as nn

from unet_diffusion_model.models.encoder import Encoder
from unet_diffusion_model.models.decoder import Decoder


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
        )

    def forward(self, input):
        # input : (batch_size, channels, height, width) = (64, 3, 32, 32)
        encoded, skip_connections = self.encoder(input)  # (64, 128, 8, 8)
        x = self.bridge(encoded)  # (64, 128, 16, 16)
        decoded = self.decoder(x, skip_connections)  # (64, 64, 32, 32)
        output = nn.Conv2d(64, 1, 1)(decoded)  # (64, 1, 32, 32)
        return output
