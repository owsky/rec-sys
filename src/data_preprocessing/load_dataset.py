import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_preprocessing.Dataset import Dataset


def load_dataset(seed: int) -> Dataset:
    # load movies metadata
    movies_df = pd.read_csv("dataset/movies.csv")

    # load tags
    tags_df = pd.read_csv("dataset/tags.csv", usecols=["movieId", "tag"])

    # load ratings
    ratings_df = pd.read_csv("dataset/ratings.csv")

    # re-index users and items
    user_id_map = {id_: i for i, id_ in enumerate(ratings_df["userId"].unique())}
    item_id_map = {id_: i for i, id_ in enumerate(ratings_df["movieId"].unique())}
    ratings_df["userId"] = ratings_df["userId"].map(user_id_map)
    ratings_df["movieId"] = ratings_df["movieId"].map(item_id_map)

    n_users = ratings_df["userId"].nunique()
    n_items = ratings_df["movieId"].nunique()

    # split ratings
    train_val_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=seed)

    tr = train_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)
    val = val_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)
    te = test_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)

    dataset = Dataset(
        ratings_df=ratings_df,
        movies_df=movies_df,
        tags_df=tags_df,
        tr=tr,
        val=val,
        te=te,
        n_users=n_users,
        n_items=n_items,
    )

    return dataset
