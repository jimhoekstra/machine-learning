import numpy as np


class OneDimClasses:

    def __init__(self, n_points_per_class, n_classes, noise_std_ratio):
        self.n_points_per_class = n_points_per_class
        self.noise_std_ratio = noise_std_ratio
        self.n_classes = n_classes
        self.noise_std = self.noise_std_ratio / self.n_classes
        self.class_centers = None

    def determine_class_centers(self):
        self.class_centers = np.linspace(-1, 1, self.n_classes)

    def generate(self):
        self.determine_class_centers()
        x = np.array([])
        y = np.array([])

        for c in range(self.n_classes):
            x = np.append(x, np.random.normal(self.class_centers[c], self.noise_std, self.n_points_per_class))
            y = np.append(y, np.ones(self.n_points_per_class) * (c+1))

        return x.reshape(-1, 1), y.reshape(-1, 1)
