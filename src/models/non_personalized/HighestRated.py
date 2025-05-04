import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from typing_extensions import override

from .NonPersonalized import NonPersonalized
from src.utils.bayesian_average import bayesian_average


class HighestRated(NonPersonalized):
    def __init__(self, train_dataset: csr_array, avg_rating_global: float):
        super().__init__(train_dataset=train_dataset, avg_rating_global=avg_rating_global)

    @override
    def _compute_scores(self, train_dataset: csr_array) -> NDArray[np.float64]:
        train_csc = self.train_dataset.tocsc()
        n_ratings = train_csc.count_nonzero(axis=0)
        # compute the average rating for each item
        ratings_sum = train_csc.sum(axis=0).ravel()
        avg_ratings = np.divide(ratings_sum, n_ratings, out=np.zeros_like(ratings_sum), where=n_ratings != 0)
        # set the threshold for the weighted average to the 75th percentile of the number of ratings
        threshold = np.percentile(n_ratings[n_ratings > 0], 75)
        # compute the bayesian average of each rating
        return bayesian_average(
            avg_ratings=avg_ratings,
            n_ratings=n_ratings,
            threshold_n_ratings=threshold,
            avg_rating_global=self.avg_rating_global,
        )
