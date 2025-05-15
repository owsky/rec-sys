from pandas import DataFrame
from typing_extensions import override

from .NonPersonalized import NonPersonalized


class MostPopular(NonPersonalized):
    """
    Class for non-personalized, most popular recommendation model
    """

    def __init__(self, train_dataset: DataFrame):
        """
        :param train_dataset: pandas DataFrame containing the training data
        """
        super().__init__(train_dataset)

    @override
    def _compute_scores(self, train_dataset: DataFrame) -> DataFrame:
        """
        Computes the non-personalized ranking scores for the training dataset according to the most popular items.
        :param train_dataset: pandas DataFrame containing the training data
        :return: pandas DataFrame containing the non-personalized ranking scores
        """
        return train_dataset.groupby("movieId").size().reset_index(name="score")
