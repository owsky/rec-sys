from typing import Optional

from loguru import logger

from src.data_preprocessing.DataLoader import DataLoader
from src.data_preprocessing.Dataset import Dataset
from src.models.collaborative_filtering.neighborhood_based.Neighborhood import Neighborhood


def train_item_item(dataset: Dataset, seed: Optional[int] = None):
    te_loader = DataLoader(dataset.te, seed=seed, batch_size=256)
    # create the model
    model = Neighborhood(train_set=dataset.sparse_tr, kind="item", similarity="pearson")
    # validate using test data loader
    rmse = model.validate_prediction(te_loader)
    logger.info(f"Item-based RMSE: {rmse}")
    ndcg = model.validate_ranking(dataset.sparse_te.toarray())
    logger.info(f"Item-based NDCG: {ndcg}")
