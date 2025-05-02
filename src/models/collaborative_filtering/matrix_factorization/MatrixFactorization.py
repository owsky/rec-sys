import numpy as np
from numpy.typing import NDArray
from src.models.Model import Model


class MatrixFactorization(Model):
    P: NDArray[np.float64]
    Q: NDArray[np.float64]
    global_bias: np.float64

    def __init__(self, n_users, n_items, n_factors):
        self.num_users = n_users
        self.num_items = n_items

        # Xavier init
        scale = 1.0 / np.sqrt(n_factors)
        self.P = np.random.normal(0, scale, size=(n_users, n_factors))
        self.Q = np.random.normal(0, scale, size=(n_items, n_factors))

        self.users_bias = np.zeros(n_users)
        self.items_bias = np.zeros(n_items)

    def predict(self, users: NDArray, items: NDArray) -> NDArray[np.float64]:
        return (
            self.global_bias
            + self.users_bias[users]
            + self.items_bias[items]
            + np.sum(self.P[users] * self.Q[items], axis=1)
        )
