import numpy as np
from numpy.typing import NDArray
from src.models.Model import Model


class MatrixFactorization(Model):
    """
    Class for matrix factorization models
    """

    P: NDArray[np.float64]
    Q: NDArray[np.float64]

    def __init__(self, n_users, n_items, n_factors):
        self.num_users = n_users
        self.num_items = n_items
        self.n_factors = n_factors

        # Xavier init for users and items factors
        scale = 1.0 / np.sqrt(n_factors)
        self.P = np.random.normal(0, scale, size=(n_users, n_factors))
        self.Q = np.random.normal(0, scale, size=(n_items, n_factors))

        self.users_bias = np.zeros(n_users)
        self.items_bias = np.zeros(n_items)
        self.global_bias = 0

    def predict(self, users: NDArray, items: NDArray) -> NDArray[np.float64]:
        """
        Predict ratings for the given users and items IDs
        :param users: numpy array of user IDs
        :param items: numpy array of item IDs
        :return: numpy array of predicted ratings
        """
        # include global, users and items biases during computation
        return (
            self.global_bias
            + self.users_bias[users]
            + self.items_bias[items]
            + np.sum(self.P[users] * self.Q[items], axis=1)
        )
