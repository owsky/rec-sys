from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from typing_extensions import override

from src.models.Model import Model


class NonPersonalized(Model, ABC):
    sorted_item_indices: NDArray[np.int64]

    def __init__(self, train_dataset: csr_array, avg_rating_global: Optional[float] = None):
        self.train_dataset = train_dataset
        self.avg_rating_global = avg_rating_global
        # convert to csc format for faster initialization
        train_csc = self.train_dataset.tocsc()
        # compute the number of rating for each item
        scores = self._compute_scores(train_dataset)
        # associated each item ID to its weighted average score
        indices = np.arange(scores.size)
        result = np.empty(indices.shape[0], dtype=[("index", int), ("score", float)])
        result["index"] = indices
        result["score"] = scores
        # sort non-ascendingly by score
        self.sorted_item_indices = np.sort(result, order="score", kind="stable")["index"][::-1]

    @override
    def top_n(self, user_index: int, n: int) -> NDArray[np.int64]:
        """
        Compute the top n recommendations for a user
        :param user_index: index of the user to recommender for
        :param n: how many items to recommend
        :return: numpy array containing the indices of the recommended items
        """
        # obtain the indices of the items already rated by the user
        user_ratings_indices = self.train_dataset[user_index].toarray().nonzero()
        # compute a mask according to the already rated items
        mask = np.isin(self.sorted_item_indices, user_ratings_indices, invert=True)
        # return the most popular, unseen items to the user
        return self.sorted_item_indices[mask][:n]

    @override
    def predict(self):
        raise RuntimeError("Content Based models cannot predict ratings")

    @abstractmethod
    def _compute_scores(self, train_dataset: csr_array) -> NDArray[np.float64]:
        pass
