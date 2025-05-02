from numpy.typing import NDArray
import numpy as np
from .mean_squared_error import mean_squared_error


def root_mean_squared_error(predictions: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> np.float64:
    """
    Compute the Root Mean Squared Error between the predictions and the ground truth.
    :param predictions: numpy array containing the predictions
    :param ground_truth: numpy array containing the ground truth
    :return: The Root Mean Squared Error between the predictions and the ground truth
    """
    # compute the mean squared error
    mse = mean_squared_error(predictions, ground_truth)
    # compute the squared root of the mean squared error
    rmse = np.sqrt(mse)
    return rmse
