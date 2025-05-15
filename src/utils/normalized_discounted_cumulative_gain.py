import numpy as np
from numpy.typing import NDArray


def ndcg_at_k(user_id: int, ranking: NDArray[np.int64], val_dataset: NDArray[np.int64], k: int) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain at k
    :param user_id: the user ID for which to compute the NDCG@k
    :param ranking: the ranking produced by the recommender system
    :param val_dataset: sparse array containing the validation dataset
    :param k: the number of items in the rankings to consider
    :return: NDCG@k
    """
    # if the ranking is empty return 0
    if len(ranking) == 0:
        return 0.0

    # if k is greater than the ranking's length, use the full ranking
    k = min(k, len(ranking))

    # only keep the top k items in the ranking
    ranking = ranking[:k]

    # extract the binary relevance scores for the items in the ranking
    relevance = np.array([val_dataset[user_id, item_id] for item_id in ranking])

    # calculate DCG
    discounts = np.log2(np.arange(1, len(relevance) + 1) + 1)
    dcg = np.sum(relevance / discounts)

    # calculate ideal DCG
    user_items = val_dataset[user_id]
    relevant_items = user_items[np.where(user_items > 0)[0]]
    sorted_relevance = np.sort(relevant_items)[::-1]
    ideal_relevance = sorted_relevance[:k]
    ideal_discounts = np.log2(np.arange(1, len(ideal_relevance) + 1) + 1)
    idcg = np.sum(ideal_relevance / ideal_discounts)

    if idcg == 0:
        return 0.0  # avoid division by zero

    return dcg / idcg
