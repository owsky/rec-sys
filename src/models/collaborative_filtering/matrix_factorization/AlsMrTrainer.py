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

    def __init__(self, model: MatrixFactorization, sparse_tr: csr_array, early_patience=5):
        """
        :param model: matrix factorization model to train
        :param sparse_tr: training data in sparse matrix format
        :param early_patience: early stopping patience
        """
        self.model = model
        self.early_patience = early_patience

        # spark initialization
        self.spark = SparkContext(
            conf=SparkConf()
            .setMaster("local")
            .setAppName("Alternating Least Squares")
            .set("spark.log.level", "ERROR")
        )

        # create and cache the ratings RDD
        tr_coo = sparse_tr.tocoo()
        self.ratings_RDD: RDD[tuple[int, int, float]] = self.spark.parallelize(
            list(zip(tr_coo.row, tr_coo.col, tr_coo.data))
        ).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_DESER)

        self.P_RDD: RDD[tuple[int, NDArray[np.float64]]] = self.spark.parallelize(
            [(u, u_weight) for u, u_weight in enumerate(self.model.P)]
        ).persist()
        self.Q_RDD: RDD[tuple[int, NDArray[np.float64]]] = self.spark.parallelize(
            [(i, i_weight) for i, i_weight in enumerate(self.model.Q)]
        ).persist()

    @override
    def training_epoch(self, reg: float):
        # unpersist last epoch’s RDDs
        self.P_RDD.unpersist()
        self.Q_RDD.unpersist()

        # update users given fixed Q_RDD
        self.P_RDD = _compute_weights(
            self.ratings_RDD, self.Q_RDD, reg, self.model.global_bias, self.model.n_factors, kind="user"
        )

        # update items given new P_RDD
        self.Q_RDD = _compute_weights(
            self.ratings_RDD, self.P_RDD, reg, self.model.global_bias, self.model.n_factors, kind="item"
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
        # after join: map to (user_id, (rating_minus_bias, item_features))
        map_fn = lambda item_id, data: (data[0][0], (data[0][1] - global_bias, data[1]))
    else:
        # prepare to join: key by user_id, carry (item_id, rating)
        join_fn = lambda user_id, item_id, rating: (user_id, (item_id, rating))
        # after join: map to (item_id, (rating_minus_bias, user_features))
        map_fn = lambda user_id, data: (data[0][0], (data[0][1] - global_bias, data[1]))

    # join each rating tuple (user_id, item_id, rating) with the corresponding fixed weight vector
    joined = ratings.map(lambda x: join_fn(*x)).join(fixed_weights)

    # remap each joined record to (key_id, (residual, features)) and group entries by key_id
    grouped = joined.map(lambda x: map_fn(*x)).groupByKey()

    def solve(entries: Iterable[tuple[float, NDArray[np.float64]]]):
        a = np.vstack([features for (_, features) in entries])
        b = np.array([residual for (residual, _) in entries])
        lhs = a.T.dot(a) + reg * np.eye(n_factors + 1)
        rhs = a.T.dot(b)
        return np.linalg.solve(lhs, rhs)

    # compute the new weight vectors for each key (user or item)
    return grouped.mapValues(solve).persist()
