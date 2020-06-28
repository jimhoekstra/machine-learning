import numpy as np


class TwoDimLine():

    def __init__(self, n_points, noise_std_ratio):
        self.n_points = n_points
        self.noise_std_ratio = noise_std_ratio
        self.MAX_SLOPE = 1

    def linear(self, slope=None):
        slopes = [slope] if slope is not None else None
        return self.poly(rank=1, slopes=slopes)

    def quadratic(self, slopes=None):
        return self.poly(rank=2, slopes=slopes)

    def qubic(self, slopes=None):
        return self.poly(rank=3, slopes=slopes)

    def poly(self, rank, slopes=None):
        x = np.arange(-1, 1, 2/self.n_points, dtype='float').reshape(-1, 1)
        slopes = slopes if slopes is not None else np.random.random(rank) * (self.MAX_SLOPE*2) - self.MAX_SLOPE

        y = np.zeros(x.shape)
        for r in range(1, rank+1):
            y += slopes[r - 1] * (x ** r)

        noise = np.random.normal(0, self.noise_std_ratio * (np.max(y) - np.min(y)), size=(self.n_points, 1))
        y += noise
        return x, y

    def periodic(self, frequency=1):
        x = np.arange(-1, 1, 2 / self.n_points, dtype='float').reshape(-1, 1)
        y = np.sin(2 * np.pi * frequency * x)

        noise = np.random.normal(0, self.noise_std_ratio * (np.max(y) - np.min(y)), size=(self.n_points, 1))
        y += noise
        return x, y