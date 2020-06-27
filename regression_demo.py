import matplotlib.pyplot as plt
from models.regression.regression import MultivariateRegressor, PolynomialRegressor
from generate_data.two_dim import TwoDimLine


def linear_regression_demo(x, y):
    linear_regressor = MultivariateRegressor()
    linear_regressor.fit(x, y)
    y_pred_linear = linear_regressor.predict(x)

    plt.plot(x, y_pred_linear, label='linear', color='darkblue')

def polynomial_regression_demo(x, y):
    quadratic_regressor = PolynomialRegressor(rank=2)
    quadratic_regressor.fit(x, y)
    y_pred_quadratic = quadratic_regressor.predict(x)

    plt.plot(x, y_pred_quadratic, label='quadratic', color='darkred')

def demo():
    data_gen = TwoDimLine(n_points=100, noise_std_ratio=0.1)
    x, y = data_gen.quadratic()
    plt.scatter(x, y, label='data', color='lightblue')

    linear_regression_demo(x, y)
    polynomial_regression_demo(x, y)

    plt.title('linear & quadratic regression')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    demo()