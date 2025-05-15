from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from typing_extensions import override

from src.models.RankingModel import RankingModel
from src.models.non_personalized.MostPopular import MostPopular
from src.utils.lists_str_join import lists_str_join


class ContentBased(RankingModel):
    """
    Class for Content Based recommendation model
    """

    def __init__(
        self,
        train_df: DataFrame,
        n_users: int,
        n_items: int,
        items_metadata: DataFrame,
        seed: Optional[int] = None,
    ):
        """
        :param train_df: DataFrame containing the training dataset
        :param n_users: number of users
        :param n_items: number of items
        :param items_metadata: pandas dataframe containing the metadata of the items
        :param seed: seed for reproducibility
        """
        self.n_users, self.n_movies = n_users, n_items
        self.items_metadata = items_metadata
        # copy the dataframe so changes don't propagate
        self.train_df = train_df.copy(deep=True)
        # make sure ratings are sorted by timestamp
        self.train_df.sort_values("timestamp", ascending=True, inplace=True)

        # train a non-personalized recommender as fallback for cold-start users
        self.np = MostPopular(train_dataset=train_df)

        # train a word vectorization model using the combined sets of genres and tags as vocabulary
        self.vec_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        vocabulary=None,  # learn the vocabulary during training
                        min_df=5,  # discard terms appearing in fewer than 5 items
                        max_df=0.5,  # discard terms appearing in more than 50% of items
                        max_features=5000,  # keep top 5k terms by TF-IDF score
                    ),
                ),
                ("svd", TruncatedSVD(n_components=100, random_state=seed)),
            ]
        )

        # build the train dataset for the word vectorization model
        vec_model_train = []
        # store a conversion map to obtain the actual movie IDs from the KNN indices
        self.knn_idx_to_movie_id = {}
        for idx, (movieId, _, genres, tags) in enumerate(items_metadata.itertuples(index=False)):
            # combine the two lists of strings in a single, space-separated string
            vec_model_train.append(lists_str_join(genres, tags))
            self.knn_idx_to_movie_id[idx] = movieId

        # train the word vectorization model and use its output to train the KNN model
        knn_train = self.vec_model.fit_transform(vec_model_train)
        # use cosine distance as metric
        self.knn_model = NearestNeighbors(metric="cosine").fit(knn_train)

        # combine the tags and genres into a single, space-separated, string
        tags_genres = items_metadata.apply(lambda row: lists_str_join(row["tags"], row["genres"]), axis=1).values
        # compute the movies' vectors using the combined tags and genres
        movies_vectors = self.vec_model.transform(tags_genres)
        # associated each movie ID to its vector
        self.movies_vectors = {
            movie_id: movie_vector for movie_id, movie_vector in zip(items_metadata["movieId"], movies_vectors)
        }

        # compute the users' profiles
        self.user_profiles = {
            user_id: profile
            for user_id in range(self.n_users)
            if (profile := self._create_user_profile(user_id=user_id)) is not None
        }

    def _create_user_profile(self, user_id: int) -> Optional[NDArray]:
        """
        Create a profile for the given user
        :param user_id: ID of the user
        :return: a numpy array containing the user's profile, None if the creation was impossible
        """
        # after how many days the rating's weight will be cut in half
        decay_half_life_days = 30
        # contribution of negative ratings
        alpha = 0.2
        # copy the user's ratings so changes do not propagate
        ratings = self.train_df[self.train_df["userId"] == user_id].copy()
        # if the user provided less than 10 ratings, consider it cold start
        if len(ratings) < 10:
            return None

        # center the ratings around the user's mean to account for rating behavior
        user_mean = ratings["rating"].mean()
        ratings["rating_centered"] = ratings["rating"] - user_mean
        # apply exponential decay to rating weights to account for time passed since rating
        now = ratings["timestamp"].max()
        ratings["dt_days"] = (now - ratings["timestamp"]) / np.timedelta64(1, "D")
        decay_const = np.log(2) / decay_half_life_days
        ratings["time_weight"] = np.exp(-decay_const * ratings["dt_days"])
        ratings["final_weight"] = ratings["rating_centered"] * ratings["time_weight"]

        # take ratings from previous 180 days
        recent = ratings[ratings["dt_days"] <= 180]

        # cap amount of ratings to 50
        if len(recent) > 50:
            recent = recent.sort_values("timestamp", ascending=False).head(50)

        # split positive and negative ratings
        pos = recent[recent["rating_centered"] > 0]
        neg = recent[recent["rating_centered"] < 0]
        # create the positive and negative profiles
        pos_profile = self._make_profile(pos)
        neg_profile = self._make_profile(neg)

        # if both the positive and the negative profiles are None, consider the user as cold start
        if pos_profile is None and neg_profile is None:
            return None
        # if only the negative profile was created, invert it
        if pos_profile is None:
            return -neg_profile
        # if only the positive profile was created, use it standalone
        if neg_profile is None:
            return pos_profile
        # otherwise compute a linear combination between the positive and the negative profiles
        return pos_profile - alpha * neg_profile

    def _make_profile(self, ratings_df: DataFrame) -> Optional[NDArray[np.float64]]:
        """
        Creates the actual user profile
        :param ratings_df: dataframe containing the user's positive/negative ratings
        :return: the numpy array containing the user's profile, None if the creation was impossible
        """
        vectors, weights = [], []
        for _, rating in ratings_df.iterrows():
            vector = self.movies_vectors.get(rating["movieId"])
            if vector is not None:
                vectors.append(vector)
                weights.append(abs(rating["final_weight"]))
        # if no vectors exist, return no profile
        if not vectors:
            return None
        weights = np.array(weights)
        # if the weights sum to 0, set them all to 1s to avoid division by zero
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        return np.average(np.vstack(vectors), axis=0, weights=weights)

    @override
    def top_n(self, users: NDArray[np.int64], n: int) -> list[NDArray[np.int64]]:
        """
        Compute the top n recommendations for the given user
        :param users: numpy array containing the user indices for which to compute the rankings
        :param n: number of recommendations to return
        :return: list of tuples of (movieId, score), with score being constant 1.0 for content based
        """
        recommendations = []
        for user_index in users:
            if user_index not in self.user_profiles.keys():
                # if the user does not have a profile, fallback to non-personalized recommender
                top_n = self.np.top_n(users=np.array([user_index]), n=n)[0]
                recommendations.append(top_n)
            else:
                # retrieve the user's profile
                user_profile = self.user_profiles[user_index]
                # add one dimension for sklearn compatibility
                user_profile = user_profile.reshape(1, -1)
                # retrieve the list of movies already rated by the user
                already_rated_ids = self.train_df[self.train_df["userId"] == user_index]["movieId"].tolist()
                # compute the number of neighbors as:
                # A = min(n + len(already_rated_ids), n_movies)
                # to always obtain n recommendations regardless of the number of ratings left by the user
                # (unless there isn't enough items in the dataset)
                # B = min(A, len(self.movies_vectors.keys())
                # to make sure that the number of neighbors never exceeds the total
                n_neighbors = min(min(n + len(already_rated_ids), self.n_movies), len(self.movies_vectors.keys()))
                # obtain the neighbors' KNN indices
                neighbors_indices = self.knn_model.kneighbors(user_profile, n_neighbors, return_distance=False)[0]
                # convert the indices back to movie IDs
                neighbors_movies_ids = [self.knn_idx_to_movie_id[knn_idx] for knn_idx in neighbors_indices]
                # remove the neighbors which the user has already interacted with and select the top N from the rest
                movie_ids = np.setdiff1d(neighbors_movies_ids, already_rated_ids, assume_unique=True)[:n]
                # store the tuple containing the recommendations and a list of constant scores
                recommendations.append(movie_ids)
        return recommendations
