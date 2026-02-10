import numpy as np


def ou_step(z, tau, sigma, dt):
    return z + (-z / tau) * dt + sigma * np.sqrt(dt) * np.random.randn()