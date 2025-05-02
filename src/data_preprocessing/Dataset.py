from dataclasses import dataclass
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Dataset:
    ratings_df: pd.DataFrame
    tr: NDArray
    val: NDArray
    te: NDArray
    movies_df: pd.DataFrame
    tags_df: pd.DataFrame
    n_users: int
    n_items: int
