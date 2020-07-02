import numpy as np


class KNN:

    def __init__(self, k):
        self.k = k
        self.features = None
        self.target = None

    def fit(self, features, target):
        self.features = features
        self.target = target

    def distances(self, x):
        x = x.T
        distances_per_feature = []
        for feature_idx in range(self.features.shape[1]):
            distances = np.abs(self.features[:, feature_idx].reshape(-1, 1) - x[feature_idx, :])
            distances_per_feature.append(distances)

        squared = [distance**2 for distance in distances_per_feature]
        norm = np.sqrt(sum(squared))
        return norm

    def predict(self, x):
        distances = self.distances(x)
        idx = np.argsort(distances, axis=0)[:self.k,:]
        classes = self.target[idx].reshape(idx.shape).astype('int')
        classes_count = [np.bincount(classes[:, i]) for i in range(classes.shape[1])]
        prediction = np.array([np.argmax(c) for c in classes_count])
        return prediction.reshape(-1, 1)
