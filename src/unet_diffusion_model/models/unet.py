### unet.py
### Implements UNet

import torch
import torch.nn as nn

from unet_diffusion_model.models.encoder import Encoder
from unet_diffusion_model.models.decoder import Decoder

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.SiLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.SiLU(),
            nn.ConvTranspose2d(1024, 1024, 2)
        )
    
    def forward(self, input):
        encoded, skip_connections = self.encoder(input)
        x = self.bridge(encoded)
        decoded = self.decoder(x, skip_connections)
        return decoded
