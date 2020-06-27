import matplotlib.pyplot as plt
from models.regression.linear_regression import LinearRegressor
from generate_data.two_dim import TwoDimData


def demo():
    data_gen = TwoDimData(n_points=100, noise_std_ratio=0.2)
    x, y = data_gen.linear()
    plt.scatter(x, y, label='data', color='lightblue')

    linear_regressor = LinearRegressor()
    linear_regressor.fit(x, y)
    y_pred = linear_regressor.predict(x)

    print(f'linear regression parameters:\n{linear_regressor.B}')
    plt.plot(x, y_pred, label='fit', color='black')
    plt.title('linear regression')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    demo()