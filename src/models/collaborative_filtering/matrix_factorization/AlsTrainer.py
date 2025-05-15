from typing import Literal

import numpy as np
from scipy.sparse import coo_array
from typing_extensions import override

from src.models.Trainer import Trainer
from .MatrixFactorization import MatrixFactorization


class AlsTrainer(Trainer):
    """
    Trainer class for Matrix factorization using alternating least squares
    """

    model: MatrixFactorization

    def __init__(self, model: MatrixFactorization, sparse_tr: coo_array, reg: float, early_patience=5):
        """
        :param model: matrix factorization model to train
        :param sparse_tr: training data in sparse matrix format
        :param reg: regularization parameter
        :param early_patience: early stopping patience
        """
        super().__init__(model=model, early_patience=early_patience)
        self.reg = reg

        def get_observed(index: int, kind: Literal["user", "item"]):
            """
            Return indices and actual values of either a user's or an item's observed ratings
            """
            row, col, data = (sparse_tr.row, sparse_tr.col, sparse_tr.data)
            if kind == "user":
                indices = np.where((row == index))[0]
                sliced_axis = col[indices]
            else:
                indices = np.where((col == index))[0]
                sliced_axis = row[indices]
            sliced_data = data[indices]
            return sliced_axis, sliced_data

        # pre-compute data for training
        self.observed_users = [get_observed(index=u, kind="user") for u in range(model.num_users)]
        self.precomputed_users_observations = {
            u: (get_observed(index=u, kind="user")) for u in range(model.num_users)
        }
        self.observed_items = [get_observed(index=i, kind="item") for i in range(model.num_items)]

    @override
    def training_epoch(self):
        """
        Training epoch for alternating least squares
        """
        # fix item factors and update user factors
        self._compute_weights(kind="user")

        # fix user factors and update item factors
        self._compute_weights(kind="item")

    def _compute_weights(self, kind: Literal["user", "item"]):
        """
        Compute the currently best weights for either users or items by fixing the others
        :param kind: whether to compute users or items weights
        """
        if kind == "user":
            obs = self.observed_users
            current_weight = self.model.P
            fixed_weight = self.model.Q
            n = self.model.num_users
        else:
            obs = self.observed_items
            current_weight = self.model.Q
            fixed_weight = self.model.P
            n = self.model.num_items

        # pre-compute regularization term multiplied to identity matrix
        reg_term = self.reg * np.eye(self.model.n_factors)

        for idx in range(n):
            # obtain the item IDs rated by the user / user IDs who rated the item
            observed_indices, observed_ratings = obs[idx]

            # subtract global mean and fixed biases
            resid = observed_ratings - self.model.global_bias

            # solve the quadratic problem
            a = fixed_weight[observed_indices, :].T @ fixed_weight[observed_indices, :] + reg_term
            b = fixed_weight[observed_indices, :].T @ resid
            current_weight[idx, :] = np.linalg.solve(a, b)
