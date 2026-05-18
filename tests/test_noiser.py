import numpy as np
from unet_diffusion_model.noiser import linear_noise_schedule, cosine_noise_schedule


def test_linear_schedule():
    betas, alpha_bars = linear_noise_schedule(1e-4, 0.02, 1000)
    assert len(betas) == 1000
    assert len(alpha_bars) == 1000
    assert abs(betas[0] - 1e-4) < 1e-9
    assert abs(betas[-1] - 0.02) < 1e-9
    assert all(alpha_bars[t+1] < alpha_bars[t] for t in range(len(alpha_bars)-1))
    assert alpha_bars[0] < 1.0
    assert alpha_bars[-1] > 0.0


def test_cosine_schedule():
    betas, alpha_bars = cosine_noise_schedule(1000)
    assert len(betas) == 1000
    assert len(alpha_bars) == 1000
    assert all(alpha_bars[t+1] < alpha_bars[t] for t in range(len(alpha_bars)-1))
    assert all(b <= 0.999 for b in betas)
    assert alpha_bars[0] > 0.999
