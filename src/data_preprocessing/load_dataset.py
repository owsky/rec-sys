import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from src.data_preprocessing.Dataset import Dataset
from src.data_preprocessing.split_by_timestamp import train_test_temporal_split


def load_dataset() -> Dataset:
    """
    Loads and transforms the dataset from the file system
    :return: an instance of the Dataset data class
    """
    # load input dataframes
    movies_df = pd.read_csv("dataset/movies.csv")
    tags_df = pd.read_csv("dataset/tags.csv", usecols=["movieId", "tag"])
    ratings_df = pd.read_csv("dataset/ratings.csv")
    ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="s")

    # set genres and tags to lowercase
    tags_df["tag"] = tags_df["tag"].str.lower()
    movies_df["genres"] = movies_df["genres"].str.lower()

    # group tags by movieId
    tags_df = (
        tags_df.groupby("movieId")["tag"]
        .agg(lambda x: list(set(x)))
        .reset_index()
        .rename(columns={"tag": "tags"})
    )

    # merge tags and movies dataframes
    metadata_df = pd.merge(movies_df, tags_df, on="movieId")

    # split genres into arrays
    metadata_df["genres"] = metadata_df["genres"].str.split("|")

    # # get complete list of movie IDs from both dataframes
    ratings_movies_ids = set(ratings_df["movieId"].unique().tolist())
    metadata_ids = set(metadata_df["movieId"].unique().tolist())
    all_movies_ids = list(ratings_movies_ids | metadata_ids)

    # re-index users and items
    user_id_map = {id_: i for i, id_ in enumerate(ratings_df["userId"].unique())}
    movie_id_map = {id_: i for i, id_ in enumerate(all_movies_ids)}
    ratings_df["userId"] = ratings_df["userId"].map(user_id_map)
    ratings_df["movieId"] = ratings_df["movieId"].map(movie_id_map)
    metadata_df["movieId"] = metadata_df["movieId"].map(movie_id_map)

    # get count of users and movies
    n_users = ratings_df["userId"].nunique()
    n_items = len(all_movies_ids)

    # split ratings
    tr_val_df, te_df = train_test_temporal_split(ratings_df, test_size=0.2)
    tr_df, val_df = train_test_temporal_split(tr_val_df, test_size=0.2)

    tr = tr_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)
    val = val_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)
    te = te_df[["userId", "movieId", "rating"]].to_numpy(dtype=np.int64)

    # create sparse matrices
    sparse_tr = csr_array((tr_df["rating"], (tr_df["userId"], tr_df["movieId"])), shape=(n_users, n_items))
    sparse_val = csr_array((val_df["rating"], (val_df["userId"], val_df["movieId"])), shape=(n_users, n_items))
    sparse_te = csr_array((te_df["rating"], (te_df["userId"], te_df["movieId"])), shape=(n_users, n_items))

    # compute average rating
    average_rating = tr_df["rating"].mean()

    dataset = Dataset(
        ratings_df=ratings_df,
        metadata_df=metadata_df,
        tr=tr,
        val=val,
        te=te,
        n_users=n_users,
        n_items=n_items,
        sparse_tr=sparse_tr,
        sparse_val=sparse_val,
        sparse_te=sparse_te,
        average_rating=average_rating,
        tr_df=tr_df,
        val_df=val_df,
        te_df=te_df,
    )

    return dataset
