import numpy as np
from numpy.typing import ArrayLike


class DataLoader:
    """
    Class for data loaders, handles batching and shuffling
    """

    def __init__(self, data: ArrayLike, batch_size: int, seed: int):
        """
        :param data: data to load
        :param batch_size: batch size
        :param seed: seed for reproducibility
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        self.rng.shuffle(self.data)
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            users = batch[:, 0]
            items = batch[:, 1]
            ratings = batch[:, 2]
            yield users, items, ratings
