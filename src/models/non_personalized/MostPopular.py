import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .NonPersonalized import NonPersonalized


class MostPopular(NonPersonalized):
    """
    Non-personalized recommender based on item popularity
    """

    def __init__(self, train_dataset: csr_array):
        """
        :param train_dataset: sparse array containing the train dataset
        """
        super().__init__(train_dataset)

    def _compute_scores(self, train_dataset: csr_array) -> NDArray[np.float64]:
        # compute how many ratings each item received, adjusted by total number of users
        n_users = train_dataset.shape[0]
        return train_dataset.tocsc().count_nonzero(axis=0) / n_users
