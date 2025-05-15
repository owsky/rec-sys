import numpy as np
from pandas import DataFrame
from typing_extensions import override

from src.utils.bayesian_average import bayesian_average
from .NonPersonalized import NonPersonalized


class HighestRated(NonPersonalized):
    """
    Class for non-personalized, highest rated recommendation model
    """

    def __init__(self, train_dataset: DataFrame, avg_rating_global: float):
        """
        :param train_dataset: pandas DataFrame containing the training data
        :param avg_rating_global: average rating across the whole training data
        """
        super().__init__(train_dataset=train_dataset, avg_rating_global=avg_rating_global)

    @override
    def _compute_scores(self, train_dataset: DataFrame) -> DataFrame:
        """
        Computes the non-personalized ranking scores for the training dataset according to the average rating
        adjusted to the number of ratings.
        :param train_dataset: pandas DataFrame containing the training data
        :return: pandas DataFrame containing the non-personalized ranking scores
        """
        count_df = train_dataset.groupby("movieId").size().reset_index(name="count")
        avg_df = train_dataset.groupby("movieId")["rating"].mean().reset_index(name="avg")
        movie_ids = count_df["movieId"].values
        n_ratings = count_df["count"].values
        avg_ratings = avg_df["avg"].values

        # set the threshold for the weighted average to the 75th percentile of the number of ratings
        threshold = np.percentile(n_ratings[n_ratings > 0], 75)
        # compute the bayesian average of each rating
        bayesian_avg = bayesian_average(
            avg_ratings=avg_ratings,
            n_ratings=n_ratings,
            threshold_n_ratings=threshold,
            avg_rating_global=self.avg_rating_global,
        )
        return DataFrame({"movieId": movie_ids, "score": bayesian_avg})
