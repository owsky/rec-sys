import pandas as pd
from scipy.sparse import csr_array

from .MatrixFactorization import MatrixFactorization


class AlsTrainer(MatrixFactorization):
    def __init__(
        self, model: MatrixFactorization, early_patience: int, n_users: int, n_items: int, n_factors: int
    ):
        super().__init__(n_users, n_items, n_factors)
        self.model = model
        self.early_patience = early_patience

    def fit(self, tr: csr_array, n_epochs: int, val: pd.DataFrame):
        pass
