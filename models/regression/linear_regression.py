import numpy as np


class LinearRegressor():

    def __init__(self):
        self.B = None

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        X = self.add_ones_to_x(X)
        self.B = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    def predict(self, X):
        X = self.add_ones_to_x(X)
        assert X.shape[1] == self.B.shape[0]
        return np.dot(X, self.B)

    @staticmethod
    def add_ones_to_x(X):
        return np.insert(X, 0, np.ones(X.shape[0]), axis=1)
