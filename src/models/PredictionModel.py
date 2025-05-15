from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.data_preprocessing import DataLoader
from src.utils.root_mean_squared_error import root_mean_squared_error


class PredictionModel(ABC):
    @abstractmethod
    def predict(self, users: NDArray[np.int64], items: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Predict ratings for the given users and items
        :param users: numpy array of user IDs
        :param items: numpy array of item IDs
        :return: numpy array of predicted ratings
        """
        pass

    def validate_prediction(self, val_loader: DataLoader) -> np.float64:
        """
        Computes the Root Mean Square Error using the given validation set
        :param val_loader: validation data loader
        :return: RMSE of the model on validation
        """
        # accumulate the validation scores for each validation batch
        validation_scores = [
            root_mean_squared_error(predictions=self.predict(users, items), ground_truth=ratings)
            for users, items, ratings in val_loader
        ]
        # return the mean of the scores
        return np.mean(validation_scores, dtype=np.float64)
