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
            nn.Conv2d(128, 256, 2),
            nn.SiLU(),
            nn.Conv2d(256, 256, 2),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 3, 2)
        )
    
    def forward(self, input):
        # input : (batch_size, channels, height, width) = (64, 3, 32, 32)
        encoded, skip_connections = self.encoder(input)  # (64, 128, 6, 6)
        x = self.bridge(encoded)  # (64, 128, 9, 9)
        decoded = self.decoder(x, skip_connections) # (64, 64, 13, 13)
        return decoded
