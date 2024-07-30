import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from src.model import Classifier

def main():
    #X, y = make_blobs(n_samples=1000, n_features=2, centers=2)
    X, y = make_circles(n_samples=1000, factor=0.2, noise=0.3)
    y = -((2*y) - 1)

    classifier = Classifier(no_linear=True, map_function="polynomial")
    classifier.train(X, y)

    x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    x2_vals = np.linspace(X[:,1].min(), X[:,1].max(), 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.contour(X1, X2, Z, levels=[0], colors="green", linewidths=3)
    plt.savefig("imagen.png")
    plt.show()


if __name__ == "__main__":
    main()