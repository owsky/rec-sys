from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.utils.normalized_discounted_cumulative_gain import ndcg_at_k


class RankingModel(ABC):

    @abstractmethod
    def top_n(self, users: NDArray[np.int64], n: int) -> NDArray[NDArray[np.int64]]:
        """
        Compute the top n recommendations for the given users
        :param users: numpy array of user IDs
        :param n: how many items to include in the users' rankings
        :return: numpy array containing the top n rankings
        """
        pass

    def validate_ranking(self, val_dataset: NDArray[NDArray[np.float64]], k=10) -> np.float64:
        """
        Computes the Normalized Discounted Cumulative Gain using the given validation set
        :param val_dataset: numpy array containing the validation data
        :param k: length of top-k recommendations to use for validation
        :return: NDCG of the model on validation
        """
        users = np.arange(val_dataset.shape[0])
        top_n_predictions = self.top_n(users=users, n=10)

        ndcgs = [
            ndcg_at_k(user_id=user_id, ranking=item_ids, val_dataset=val_dataset, k=k)
            for user_id, item_ids in zip(users, top_n_predictions)
            if len(item_ids) > 0
        ]

        return np.mean(ndcgs)
