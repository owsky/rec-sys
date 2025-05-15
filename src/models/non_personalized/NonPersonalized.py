from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame
from typing_extensions import override

from src.models.RankingModel import RankingModel


class NonPersonalized(RankingModel, ABC):
    """
    Abstract class for non-personalized ranking models.
    """

    def __init__(self, train_dataset: DataFrame, avg_rating_global: Optional[float] = None):
        """
        :param train_dataset: pandas DataFrame containing the training data
        :param avg_rating_global: average rating across the whole training data
        """
        self.train_dataset = train_dataset
        self.avg_rating_global = avg_rating_global or 0.0
        # compute the number of rating for each item
        scores_df = self._compute_scores(train_dataset)
        # sort non-ascendingly by score
        scores_df.sort_values(by=["score"], ascending=False, inplace=True, kind="stable")
        self.sorted_item_indices = scores_df["movieId"].values

    @override
    def top_n(self, users: ArrayLike, n: int) -> list[list[int]]:
        """
        Compute the top n recommendations for a user
        :param users: numpy array containing the user indices for which to compute the rankings
        :param n: how many items to recommend
        :return: numpy array containing the indices of the recommended items
        """
        recommendations = []
        for user in users:
            # obtain the indices of the items already rated by the user
            user_ratings_indices = self.train_dataset[self.train_dataset["userId"] == user]["movieId"].values
            unrated_items = np.setdiff1d(self.sorted_item_indices, user_ratings_indices, assume_unique=True)
            recommendations.append(unrated_items[:n])
        return recommendations

    @abstractmethod
    def _compute_scores(self, train_dataset: DataFrame) -> DataFrame:
        """
        Computes the non-personalized ranking scores for the training dataset.
        :param train_dataset: pandas DataFrame containing the training data
        :return: pandas DataFrame containing the non-personalized ranking scores
        """
        pass
