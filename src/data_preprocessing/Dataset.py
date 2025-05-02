from dataclasses import dataclass
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_array


@dataclass
class Dataset:
    """
    Data class storing the dataset's components needed for training the models
    """

    ratings_df: pd.DataFrame
    tr: NDArray
    val: NDArray
    te: NDArray
    movies_df: pd.DataFrame
    tags_df: pd.DataFrame
    n_users: int
    n_items: int
    sparse_tr: csr_array
