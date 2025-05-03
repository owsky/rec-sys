from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.models.Model import Model


class MatrixFactorization(Model):
    """
    Class for matrix factorization models
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int, average_rating: float, seed: Optional[int]):
        self.num_users = n_users
        self.num_items = n_items
        self.n_factors = n_factors

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Xavier init for users and items factors
        scale = 1.0 / np.sqrt(n_factors + 1)
        self.P = rng.normal(0, scale, size=(n_users, n_factors + 1))
        self.Q = rng.normal(0, scale, size=(n_items, n_factors + 1))
        # initialize biases within embeddings
        self.P[:, 0] = 0
        self.Q[:, 0] = 0
        self.global_bias = average_rating

    def predict(self, users: NDArray, items: NDArray) -> NDArray[np.float64]:
        """
        Predict ratings for the given users and items IDs
        :param users: numpy array of user IDs
        :param items: numpy array of item IDs
        :return: numpy array of predicted ratings
        """
        # include global, users and items biases during computation
        return self.global_bias + np.sum(self.P[users] * self.Q[items], axis=1)
