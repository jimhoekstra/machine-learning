# import matplotlib.pyplot as plt
from models.regression import MultivariateRegressor, PolynomialRegressor
from metrics.regression import mse
from data.regression import Line


def linear_regression_demo(x, y):
    linear_regressor = MultivariateRegressor()
    linear_regressor.fit(x, y)
    y_pred = linear_regressor.predict(x)
    print(f'MSE linear model: {mse(y, y_pred)}')

    # plt.plot(x, y_pred, label='linear', color='darkblue')


def polynomial_regression_demo(x, y, poly_rank):
    quadratic_regressor = PolynomialRegressor(rank=poly_rank)
    quadratic_regressor.fit(x, y)
    y_pred = quadratic_regressor.predict(x)
    print(f'MSE polynomial model: {mse(y, y_pred)} for rank {poly_rank}')

    # plt.plot(x, y_pred, label='polynomial', color='darkred')


def demo(x, y, poly_rank):
    # plt.scatter(x, y, label='data', color='lightblue')

    linear_regression_demo(x, y)
    polynomial_regression_demo(x, y, poly_rank)

    # plt.title('linear & polynomial regression')
    # plt.legend()
    # plt.show()


def main():
    line_gen = Line(n_points=200, noise_std_ratio=0.2)

    print('Quadratic data')
    x, y = line_gen.quadratic()
    demo(x, y, poly_rank=2)

    print('\nPeriodic data')
    x, y = line_gen.periodic()
    demo(x, y, poly_rank=6)


if __name__ == '__main__':
    main()
