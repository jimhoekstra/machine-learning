import numpy as np


class Circles:

    def __init__(self, n_points_per_class, n_classes, noise_std):
        self.n_points_per_class = n_points_per_class
        self.n_classes = n_classes
        self.noise_std = noise_std

    def generate(self):
        x = np.array([])
        y = np.array([])
        target = np.array([])

        for c in range(self.n_classes):
            rad = np.linspace(0, 2*np.pi, self.n_points_per_class)
            x = np.append(x, (c+1) * np.sin(rad) + np.random.normal(0, self.noise_std, rad.shape))
            y = np.append(y, (c+1) * np.cos(rad) + np.random.normal(0, self.noise_std, rad.shape))
            target = np.append(target, np.ones(self.n_points_per_class) * (c+1))

        features = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        target = target.reshape(-1, 1)
        return features, target
