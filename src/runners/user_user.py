from typing import Optional

from loguru import logger

from src.data_preprocessing import Dataset
from src.data_preprocessing.DataLoader import DataLoader
from src.models.collaborative_filtering.neighborhood_based.Neighborhood import Neighborhood


def train_user_user(dataset: Dataset, seed: Optional[int] = None):
    te_loader = DataLoader(dataset.te, seed=seed, batch_size=256)
    # create the model
    model = Neighborhood(train_set=dataset.sparse_tr, kind="user", similarity="pearson")
    # validate using test data loader
    rmse = model.validate_prediction(te_loader)
    logger.info(f"User-based RMSE: {rmse}")
    ndcg = model.validate_ranking(dataset.sparse_te.toarray())
    logger.info(f"User-based NDCG: {ndcg}")
