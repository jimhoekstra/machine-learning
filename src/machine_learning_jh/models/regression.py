import numpy as np


class MultivariateRegressor:

    def __init__(self, method='ls', lr=0.001, max_iter=10000):
        self.B = None
        self.method = method
        self.lr = lr
        self.max_iter = max_iter
        self.convergence_threshold = 1e-5

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        X = self.add_ones_to_x(X)
        if self.method == 'ls':
            self.fit_ls(X, Y)
        elif self.method == 'gd':
            self.fit_gd(X, Y)

    def fit_ls(self, X, Y):
        self.B = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    def fit_gd(self, X, y):
        self.B = np.random.normal(0, 0.1, (X.shape[1], 1))

        prev_loss = 0.0
        for i in range(self.max_iter):
            y_pred = np.dot(X, self.B)
            error = y - y_pred

            loss = np.mean(error ** 2)
            if np.abs(prev_loss - loss) < self.convergence_threshold:
                break

            prev_loss = loss
            grad = np.dot(X.T, 2*error)
            self.B += (self.lr * grad)


    def predict(self, X):
        X = self.add_ones_to_x(X)
        assert X.shape[1] == self.B.shape[0]
        return np.dot(X, self.B)

    @staticmethod
    def add_ones_to_x(X):
        return np.insert(X, 0, np.ones(X.shape[0]), axis=1)


class PolynomialRegressor(MultivariateRegressor):

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
