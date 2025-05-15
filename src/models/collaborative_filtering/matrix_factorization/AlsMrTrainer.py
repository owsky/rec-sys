from typing import Literal, Iterable

import numpy as np
from numpy.typing import NDArray
from pyspark import RDD, SparkContext, SparkConf, StorageLevel
from scipy.sparse import csr_array
from typing_extensions import override

from src.models.Trainer import Trainer
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization


class AlsMrTrainer(Trainer):
    """
    Trainer class for Matrix factorization using Alternating Least Squares on PySpark
    """

    model: MatrixFactorization

    def __init__(self, model: MatrixFactorization, sparse_tr: csr_array, reg: float, early_patience=5):
        """
        :param model: matrix factorization model to train
        :param sparse_tr: training data in sparse matrix format
        :param reg: regularization parameter
        :param early_patience: early stopping patience
        """
        super().__init__(model=model, early_patience=early_patience)
        self.reg = reg

        # spark initialization
        self.spark = SparkContext(
            conf=SparkConf()
            .setMaster("local")
            .setAppName("Alternating Least Squares")
            .set("spark.log.level", "ERROR")
        )

        # create and cache the ratings' RDD
        tr_coo = sparse_tr.tocoo()
        self.ratings_RDD: RDD[tuple[int, int, float]] = self.spark.parallelize(
            list(zip(tr_coo.row, tr_coo.col, tr_coo.data))
        ).persist(
            storageLevel=StorageLevel.MEMORY_AND_DISK_DESER  # type: ignore
        )
        # create and cache the embeddings' RDDs
        self.P_RDD: RDD[tuple[int, NDArray[np.float64]]] = self.spark.parallelize(
            [(u, u_weight) for u, u_weight in enumerate(self.model.P)]
        ).persist(
            storageLevel=StorageLevel.MEMORY_AND_DISK_DESER  # type: ignore
        )
        self.Q_RDD: RDD[tuple[int, NDArray[np.float64]]] = self.spark.parallelize(
            [(i, i_weight) for i, i_weight in enumerate(self.model.Q)]
        ).persist(
            storageLevel=StorageLevel.MEMORY_AND_DISK_DESER  # type: ignore
        )

    @override
    def training_epoch(self):
        """
        Training epoch: compute the updated embeddings in RDD form
        """
        # unpersist last epoch’s users' RDD as it's no longer needed
        self.P_RDD.unpersist()

        # update users given fixed Q_RDD
        self.P_RDD = _compute_weights(
            ratings=self.ratings_RDD,
            fixed_weights=self.Q_RDD,
            reg=self.reg,
            global_bias=self.model.global_bias,
            n_factors=self.model.n_factors,
            kind="user",
        ).persist(
            storageLevel=StorageLevel.MEMORY_AND_DISK_DESER  # type: ignore
        )
        self.Q_RDD.unpersist()

        # update items given new P_RDD
        self.Q_RDD = _compute_weights(
            ratings=self.ratings_RDD,
            fixed_weights=self.P_RDD,
            reg=self.reg,
            global_bias=self.model.global_bias,
            n_factors=self.model.n_factors,
            kind="item",
        ).persist(
            storageLevel=StorageLevel.MEMORY_AND_DISK_DESER  # type: ignore
        )

    @override
    def after_training(self):
        # clean up the RDD
        self.ratings_RDD.unpersist()
        # stop the spark process
        self.spark.stop()


def _compute_weights(
    ratings: RDD[tuple[int, int, float]],
    fixed_weights: RDD[tuple[int, NDArray[np.float64]]],
    reg: float,
    global_bias: float,
    n_factors: int,
    kind: Literal["user", "item"],
) -> RDD[tuple[int, NDArray[np.float64]]]:
    """
    Computes the new non-fixed weights using the fixed factor
    :param ratings: RDD containing (u,i,r) triples
    :param fixed_weights: factor to keep as fixed for this optimization round
    :param reg: regularization hyperparameter
    :param global_bias: global bias, average rating across the dataset
    :param n_factors: number of factors for the matrix factorization
    :param kind: whether to compute the users' or the items' weights
    :return: a new RDD containing the optimized weights
    """
    if kind == "user":
        # prepare to join: key by item_id, carry (user_id, rating)
        join_fn = lambda user_id, item_id, rating: (item_id, (user_id, rating))

        def map_fn(_item_id, data):
            user_id, rating = data[0]
            item_features = data[1]
            return user_id, (rating - global_bias, item_features)

    else:
        # prepare to join: key by user_id, carry (item_id, rating)
        join_fn = lambda user_id, item_id, rating: (user_id, (item_id, rating))

        def map_fn(_user_id, data):
            item_id, rating = data[0]
            user_features = data[1]
            return item_id, (rating - global_bias, user_features)

    # join each rating tuple (user_id, item_id, rating) with the corresponding fixed weight vector
    joined = ratings.map(lambda x: join_fn(*x)).join(fixed_weights)

    # remap each joined record to (key_id, (residual, features)) and group entries by key_id
    grouped = joined.map(lambda x: map_fn(*x)).groupByKey()

    def solve(entries: Iterable[tuple[float, NDArray[np.float64]]]) -> NDArray[NDArray[np.float64]]:
        a = np.vstack([features for (_, features) in entries])
        b = np.array([residual for (residual, _) in entries])
        lhs = a.T.dot(a) + reg * np.eye(n_factors)
        rhs = a.T.dot(b)
        return np.linalg.solve(lhs, rhs)

    # compute the new weight vectors for each key (user or item)
    return grouped.mapValues(solve)
