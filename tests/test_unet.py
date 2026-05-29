from unet_diffusion_model.models.encoder import EncoderSubBlock, Encoder
from unet_diffusion_model.models.decoder import DecoderSubBlock, Decoder
from unet_diffusion_model.models.unet import UNet

import torch


def test_encoder_sub_block():
    encoder_block = EncoderSubBlock(3, 64)
    tensor = torch.randn(64, 3, 32, 32)
    output = encoder_block(tensor)
    assert output.shape == torch.Size([64, 64, 32, 32])


def test_encoder():
    encoder = Encoder()
    tensor = torch.randn(64, 3, 32, 32)
    output, skips = encoder(tensor)
    assert output.shape == torch.Size([64, 128, 8, 8])
    assert len(skips) == 2
    assert skips[0].shape == torch.Size([64, 64, 32, 32])
    assert skips[1].shape == torch.Size([64, 128, 16, 16])


def test_decoder_sub_block():
    decoder_block = DecoderSubBlock(128, 64)
    tensor = torch.randn(64, 128, 32, 32)
    output = decoder_block(tensor)
    assert output.shape == torch.Size([64, 64, 32, 32])


def test_decoder():
    decoder = Decoder()
    encoded = torch.randn(64, 128, 16, 16)
    skips = [torch.randn(64, 64, 32, 32), torch.randn(64, 128, 16, 16)]
    output = decoder(encoded, skips)
    assert output.shape == torch.Size([64, 64, 32, 32])


def test_unet():
    model = UNet()
    tensor = torch.randn(64, 3, 32, 32)
    output = model(tensor)
    assert output.shape == torch.Size([64, 2, 32, 32])

    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    assert not torch.isnan(output).any()
    assert output.dtype == torch.float32
    assert output.device == tensor.device
