import numpy as np
from math import prod

def linear_noise_schedule(b_start: float, b_end: float, T: int):
    betas = [b_start + (t/(T-1))*(b_end-b_start) for t in range(T)]
    # betas = np.linspace(b_start, b_end, T)
    alphas = [1-b for b in betas]
    alpha_bars = [prod(alphas[:t+1]) for t in range(T)]
    return betas, alpha_bars


def cosine_noise_schedule(T: int, s: float=0.008):
    def f(t):
        return float(np.cos(((t/T + s)/(1+s))*(np.pi/2))**2)
    f0 = f(0)
    alpha_bars = [f(t)/f0 for t in range(T)]
    alphas = alpha_bars[:1] + [alpha_bars[t+1]/alpha_bars[t] for t in range(T-1)]
    # Prevent beta from being >0.999 near t=T
    betas = [min(1 - a, 0.999) for a in alphas]
    return betas, alpha_bars
