import numpy as np


def add_ones_to_x(X):
    return np.insert(X, 0, np.ones(X.shape[0]), axis=1)


class MultivariateRegressor():

    def __init__(self):
        self.B = None

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        X = add_ones_to_x(X)
        self.B = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    def predict(self, X):
        X = add_ones_to_x(X)
        assert X.shape[1] == self.B.shape[0]
        return np.dot(X, self.B)


class PolynomialRegressor():

    def __init__(self, rank):
        self.B = None
        self.rank = rank

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        X = add_ones_to_x(X)
        X = self.add_ranks_to_data(X)
        self.B = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    def predict(self, X):
        X = add_ones_to_x(X)
        X = self.add_ranks_to_data(X)
        assert X.shape[1] == self.B.shape[0]
        return np.dot(X, self.B)

    def add_ranks_to_data(self, X):
        for r in range(2, self.rank+1):
            X = np.insert(X, r, X[:,r-1] * X[:, 1], axis=1)
        return X
