import numpy as np
from numpy.typing import NDArray


def bayesian_average(
    avg_ratings: NDArray[np.float64],
    n_ratings: NDArray[np.int64],
    threshold_n_ratings: float,
    avg_rating_global: float,
) -> NDArray[np.float64]:
    """
    Adjust the given average ratings according to the Bayesian average method, which rescales them according
    to how many ratings were given for the specific item.
    :param avg_ratings: numpy array containing the average ratings for the items
    :param n_ratings: numpy array containing the number of ratings for the items
    :param threshold_n_ratings: threshold to decide inclusion of the items
    :param avg_rating_global: global average rating
    :return: the adjusted average ratings
    """
    rescaled_n_ratings = n_ratings / (n_ratings + threshold_n_ratings)
    rescaled_threshold = threshold_n_ratings / (n_ratings + threshold_n_ratings)
    return rescaled_n_ratings * avg_ratings + rescaled_threshold * avg_rating_global
