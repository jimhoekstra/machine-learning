import numpy as np
from .linear_regression import LinearRegression


class PolynomialRegression(LinearRegression):

    def __init__(self, rank, method='ls', lr=0.001, max_iter=10000):
        super().__init__(method, lr, max_iter)
        self.B = None
        self.rank = rank
        self.method = method
        self.lr = lr
        self.max_iter = int(max_iter)

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        X = self.add_ones_to_x(X)
        X = self.add_ranks_to_data(X)
        if self.method == 'ls':
            self.fit_ls(X, Y)
        elif self.method == 'gd':
            self.fit_gd(X, Y)

    def predict(self, X):
        X = self.add_ones_to_x(X)
        X = self.add_ranks_to_data(X)
        assert X.shape[1] == self.B.shape[0]
        return np.dot(X, self.B)

    def add_ranks_to_data(self, X):
        for r in range(2, self.rank+1):
            X = np.insert(X, r, X[:,r-1] * X[:, 1], axis=1)
        return X
