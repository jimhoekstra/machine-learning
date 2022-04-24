# import matplotlib.pyplot as plt
# import numpy as np
from machine_learning_jh.data import OneDimClasses, Circles
from machine_learning_jh.models import KNN


def one_dim_demo(n_points, n_classes):
    data_gen = OneDimClasses(n_points, n_classes, 1.5)
    features, target = data_gen.generate()

    knn = KNN(k=3)
    knn.fit(features, target)
    prediction = knn.predict(features)

    # plt.figure(figsize=(8, 5))
    # scatter = plt.scatter(features, target, c=prediction)

    # plt.yticks(np.arange(n_classes) + 1)
    # plt.legend(*scatter.legend_elements(), title="Class")
    # plt.title('y axis shows true class, color shows predicted class')
    # plt.show()


def circle_demo(n_points, n_classes):
    data_gen = Circles(n_points, n_classes, 0.5)
    features, target = data_gen.generate()

    knn = KNN(k=3)
    knn.fit(features, target)
    predictions = knn.predict(features)

    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(features[:, 0], features[:, 1], target, c=predictions)

    # ax.set_zticks(np.arange(n_classes) + 1)
    # plt.legend(*scatter.legend_elements(), title="Class")
    # plt.title('z axis shows true class, color shows predicted class')
    # plt.show()


def main():
    n_points = 100
    n_classes = 3
    one_dim_demo(n_points, n_classes)
    circle_demo(n_points, n_classes)


if __name__ == '__main__':
    main()
