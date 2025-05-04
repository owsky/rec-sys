from abc import ABC, abstractmethod

import numpy as np

from src.DataLoader import DataLoader
from src.utils.root_mean_squared_error import root_mean_squared_error


class Model(ABC):
    """
    Abstract class that represents a generic machine learning model
    """

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def top_n(self, *args, **kwargs):
        pass

    def validate(self, val_loader: DataLoader) -> np.float64:
        """
        Validate the model using the given validation data loader
        :param val_loader: validation data loader
        :return: validation score
        """
        validation_scores = []
        # accumulate the validation scores for each validation batch
        for users, items, ratings in val_loader:
            preds = self.predict(users, items)
            rmse = root_mean_squared_error(preds, ratings)
            validation_scores.append(rmse)
        # return the mean of the scores
        return np.mean(validation_scores, dtype=np.float64)
