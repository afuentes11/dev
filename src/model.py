import numpy as np
from src.utils import zip_shuffled

def loss(z):
    """
    Calculates the hinge loss.

    Parameters:
    -----------
    z : float
        The product of the label and the classifier's decision function output.

    Returns:
    --------
    float
        The hinge loss value.
    """
    return np.maximum(0, 1 - z)

class Classifier():
    """
    A binary classifier that can optionally use non-linear mapping functions.

    Attributes:
    -----------
    no_linear : bool
        Indicates whether to use non-linear mapping for the input points.
    theta : np.ndarray
        The weight vector for the classifier.
    theta0 : float
        The bias term for the classifier.
    map_points : function
        A function that maps input points to a higher-dimensional space.

    Methods:
    --------
    train(points, labels, learning_factor=0.01, L=0.01, n_epochs=1000)
        Trains the classifier using the provided points and labels.
    predict(points)
        Predicts the labels for the given points.
    """
    def __init__(self, no_linear=False, map_function="binomial") -> None:
        """
        Initializes the Classifier with optional non-linear mapping.

        Parameters:
        -----------
        no_linear : bool, optional
            If True, the input points will be mapped to a higher-dimensional space using the specified map_function. Default is False.
        map_function : str, optional
            The mapping function to use if no_linear is True. Can be "binomial" or "polynomial". Default is "binomial".

        Raises:
        -------
        ValueError
            If an unsupported map_function is provided.
        """
        self.no_linear = no_linear
        self.theta = 0
        self.theta0 = 0

        match map_function:
            case "binomial":
                self.map_points = lambda points: np.array([[x1**2, np.sqrt(2)*x1*x2, x2**2] for x1, x2 in points], dtype=np.float64)
            case "polynomial":
                self.map_points = lambda points: np.array([[x1, x2, x1**2, np.sqrt(2)*x1*x2, x2**2, x1**3, np.sqrt(3)*(x1**2)*x2, np.sqrt(3)*x1*(x2**2), x2**3, 1] for x1, x2 in points], dtype=np.float64)
            case _:
                raise ValueError(f"""Classifier: Map function "{map_function}" not found. Use "binomial" or "polynomial" """)
    
    def train(self, points, labels, learning_factor=0.01, L=0.01, n_epochs=1000):
        """
        Trains the classifier using the provided points and labels.

        Parameters:
        -----------
        points : array-like
            The input points for training. Each point should be a list or array of features.
        labels : array-like
            The labels corresponding to the input points. Each label should be +1 or -1.
        learning_factor : float, optional
            The learning rate for the gradient descent algorithm. Default is 0.01.
        L : float, optional
            The regularization parameter. Default is 0.01.
        n_epochs : int, optional
            The number of epochs for training. Default is 1000.
        """

        points = self.map_points(points) if self.no_linear else points
        self.theta = np.zeros_like(points[0])
        for _ in range(n_epochs):
            for x_i, y_i in zip_shuffled(points, labels):
                z = y_i * ((self.theta @ x_i) + self.theta0)
                if loss(z) > 0:
                    self.theta = (1 - learning_factor*L)*self.theta + (learning_factor*y_i*x_i)
                    self.theta0 += learning_factor * y_i
                else:
                    self.theta = (1 - learning_factor*L)*self.theta
                    self.theta0 = self.theta0

    def predict(self, points):
        """
        Predicts the labels for the given points.

        Parameters:
        -----------
        points : array-like
            The input points to predict. Each point should be a list or array of features.

        Returns:
        --------
        np.ndarray
            The predicted labels for the input points.
        """
        points = self.map_points(points) if self.no_linear else points
        return np.sign(np.dot(points, self.theta) + self.theta0)

