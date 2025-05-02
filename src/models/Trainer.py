from abc import abstractmethod, ABC


class Trainer(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
