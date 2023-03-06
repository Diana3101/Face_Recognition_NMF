import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt


def data_load():
    lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.3)
    X = lfw_people.data
    y = lfw_people.target

    n,h,w = lfw_people.images.shape

    # normalize X from [0, 255] to [0, 1]
    X = X / 255.0
    return X, y, w, h


if __name__ == "__main__":
    X, y, w, h = data_load()

    n_row = 3
    n_col = 4
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(X[i+5].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()

    print(X.shape)
    print(y.shape)

    print(np.unique(y, return_counts=True))
