from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing_extensions import override

from ...PredictionModel import PredictionModel
from ...RankingModel import RankingModel


class Neighborhood(PredictionModel, RankingModel):
    """
    Class for Memory-based Collaborative Filtering. Supports both User-based and Item-based recommendation, with
    either Pearson Correlation of Adjusted Cosine Similarity.
    """

    def __init__(
        self, train_set: csr_array, kind: Literal["user", "item"], similarity: Literal["pearson", "cosine"]
    ):
        """
        :param train_set: sparse array containing the ratings
        :param kind: whether to compute user-user or item-item recommendation
        :param similarity: whether to use pearson correlation or adjusted cosine similarity
        """
        self.kind = kind
        self.similarity = similarity
        self.train_set = train_set.toarray()

        self.means = self._compute_biases()
        self.similarities = self._compute_similarities()

    def _compute_biases(self) -> NDArray[np.float64]:
        """
        If self.kind is "user", compute biases for items;
        If self.kind is "item", compute biases for users;
        :return: numpy array containing the biases
        """
        axis = 0 if self.kind == "user" else 1
        # count the number of ratings across the axis
        denominator = np.count_nonzero(self.train_set, axis=axis)
        denominator[denominator == 0] = 1  # avoid division by zero
        # compute the biases as the means of the ratings
        return self.train_set.sum(axis=axis) / denominator

    def _compute_similarities(self) -> NDArray[NDArray[np.float64]]:
        """
        Computes the similarity matrix between the users/items
        """
        ratings = self.train_set if self.kind == "user" else self.train_set.T
        if self.similarity == "pearson":
            with np.errstate(divide="ignore", invalid="ignore"):
                similarities = np.corrcoef(ratings)
                similarities[np.isnan(similarities)] = 0
                return similarities
        else:
            # center ratings according to the means to obtain adjusted cosine similarity
            adjusted_ratings = ratings - self.means
            return cosine_similarity(adjusted_ratings)

    def _predict_pair(self, user: int, item: int, k=3) -> np.float64:
        """
        Predicts the rating for the given pair of user and item.
        :param user: user ID
        :param item: item ID
        :param k: how many neighbors to consider
        :return: the predicted rating
        """
        if self.kind == "user":
            # neighbor users who rated item i
            indices = np.where(self.train_set[:, item] != 0)[0]
            index = user
            index_mean = self.means[item]
        else:
            # neighbor items rated by user u
            indices = np.where(self.train_set[user, :] != 0)[0]
            index = item
            index_mean = self.means[user]

        # get similarities
        sims = self.similarities[index, indices]
        # pick top-k neighbors
        idx = np.argsort(-np.abs(sims))[:k]
        neighbors = indices[idx]
        weights = sims[idx]

        ratings_indices = (neighbors, item) if self.kind == "user" else (user, neighbors)

        # compute numerator
        num = np.sum(weights * (self.train_set[*ratings_indices] - index_mean))
        den = np.sum(np.abs(weights)) or 1.0
        return index_mean + num / den

    @override
    def predict(self, users: NDArray[np.int64], items: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Predict ratings for the given users and items
        :param users: numpy array of user IDs
        :param items: numpy array of item IDs
        :return: numpy array of predicted ratings
        """
        return np.array([self._predict_pair(user, item) for user, item in zip(users, items)])

    @override
    def top_n(self, users: NDArray[np.int64], n: int) -> NDArray[NDArray[np.int64]]:
        """
        Compute the top n recommendations for the given users
        :param users: numpy array of user IDs
        :param n: how many items to include in the users' rankings
        :return: numpy array containing the top n rankings
        """
        items = np.arange(self.train_set.shape[1])
        # compute the predictions in parallel
        preds = Parallel(n_jobs=-1, backend="loky")(
            delayed(lambda u: np.array([self._predict_pair(u, i) for i in items]))(u)
            for u in tqdm(users, desc="Computing top-n recommendations...", dynamic_ncols=True, leave=False)
        )
        preds = np.array(preds)
        return np.argsort(-preds, axis=1)[:, :n]
