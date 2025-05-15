from typing import Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from src.models.PredictionModel import PredictionModel
from src.models.RankingModel import RankingModel


class MatrixFactorization(PredictionModel, RankingModel):
    """
    Class for matrix factorization models
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int, average_rating: float, seed: Optional[int]):
        """
        :param n_users: number of users
        :param n_items: number of items
        :param n_factors: number of latent factors to use during training
        :param average_rating: global average rating
        :param seed: seed for reproducibility
        """
        self.num_users = n_users
        self.num_items = n_items
        self.n_factors = n_factors

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Xavier init for users and items factors
        scale = 1.0 / np.sqrt(n_factors + 1)
        self.P = rng.normal(0, scale, size=(n_users, n_factors))
        self.Q = rng.normal(0, scale, size=(n_items, n_factors))
        # initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = average_rating

    @override
    def predict(self, users: NDArray[np.int64], items: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Predict ratings for the given users and items IDs
        :param users: numpy array of user IDs
        :param items: numpy array of item IDs
        :return: numpy array of predicted ratings
        """
        # include global, users and items biases during computation
        return (
            self.global_bias
            + self.user_biases[users]
            + self.item_biases[items]
            + np.sum(self.P[users] * self.Q[items], axis=1)
        )

    @override
    def top_n(self, users: NDArray[np.int64], n: int) -> NDArray[NDArray[np.int64]]:
        """
        Compute the top n recommendations for the given users
        :param users: numpy array containing the user IDs to recommend for
        :param n: how many recommendations to return
        :return: 2D numpy array containing the indices of the top n recommendations
        """
        items = np.arange(self.num_items)
        # perform elementwise multiplication
        preds = self.global_bias + (self.P[users] @ self.Q[items].T)
        # retrieve the indices of the highest scoring movies
        return np.argsort(-preds, axis=1)[:, :n]
