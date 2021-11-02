import numpy as np


def rickerwave(freq: float, dt: float):
    '''
    Generates ricker wavelet with frequency freq and sample rate dt. 
    '''
    def ricker(t):
        return (1 - 2 * t**2) * np.exp(-t**2)

    assert freq < 0.2 * 1.0 / (2.0 * dt)
    length = 2 * (2.2 / (freq * dt)) // 2
    t = (np.pi * freq * dt) * np.arange(-length // 2, length // 2 + 1)
    return ricker(t)
