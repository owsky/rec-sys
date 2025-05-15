from dataclasses import dataclass

from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import csr_array


@dataclass
class Dataset:
    """
    Data class storing the dataset's components needed for training the models
    """

    ratings_df: DataFrame
    tr: NDArray
    val: NDArray
    te: NDArray
    tr_df: DataFrame
    val_df: DataFrame
    te_df: DataFrame
    metadata_df: DataFrame
    n_users: int
    n_items: int
    sparse_tr: csr_array
    sparse_val: csr_array
    sparse_te: csr_array
    average_rating: float
