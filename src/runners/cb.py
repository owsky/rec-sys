from typing import Optional

from loguru import logger

from src.data_preprocessing.Dataset import Dataset
from src.models.content_based.ContentBased import ContentBased


def train_content_based(dataset: Dataset, seed: Optional[int] = None):
    model = ContentBased(
        train_df=dataset.tr_df,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        items_metadata=dataset.metadata_df,
        seed=seed,
    )
    ndcg = model.validate_ranking(val_dataset=dataset.sparse_te.toarray())
    logger.info(f"Content-based NDCG: {ndcg}")
