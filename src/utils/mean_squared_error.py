import numpy as np
from numpy.typing import NDArray


def mean_squared_error(predictions: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> np.float64:
    """
    Compute the Mean Squared Error between the predictions and the ground truth.
    :param predictions: numpy array containing the predictions
    :param ground_truth: numpy array containing the ground truth
    :return: the Mean Squared Error between the predictions and the ground truth
    """
    # compute the difference between the predictions and the ground truth
    difference = np.subtract(predictions, ground_truth)
    # square it so the sign is ignored and big differences are given more weight
    squared_difference = np.square(difference)
    # compute the mean
    mse = np.mean(squared_difference, dtype=np.float64)
    return mse
