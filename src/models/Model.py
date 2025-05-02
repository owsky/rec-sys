from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class that represents a generic machine learning model
    """

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
