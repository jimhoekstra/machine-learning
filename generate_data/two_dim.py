import numpy as np


class TwoDimData():

    def __init__(self, n_points, noise_std_ratio):
        self.n_points = n_points
        self.noise_std_ratio = noise_std_ratio
        self.noise_std = self.n_points * self.noise_std_ratio
        self.MAX_SLOPE = 3

    def linear(self, slope=None):
        x = np.arange(1, self.n_points+1, dtype='float').reshape(-1, 1)
        slope = slope if slope is not None else np.random.random() * self.MAX_SLOPE
        noise = np.random.normal(0, self.noise_std, size=(self.n_points, 1))
        y = slope * x + noise
        return x, y