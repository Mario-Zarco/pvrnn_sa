import numpy as np


def ou_step(z, tau, sigma, dt):
    return z + (-z / tau) * dt + sigma * np.sqrt(dt) * np.random.randn()


def ou_update(x, mu, tau, sigma, dt):
    exp_term = np.exp(-dt / tau)
    variance = sigma**2 * (tau / 2.0) * (1 - exp_term**2)
    return (
        mu
        + (x - mu) * exp_term
        + np.sqrt(variance) * np.random.randn()
    )
