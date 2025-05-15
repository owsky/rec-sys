from loguru import logger

from src.data_preprocessing.Dataset import Dataset
from src.models.non_personalized.HighestRated import HighestRated


def train_highest_rated(dataset: Dataset):
    hr = HighestRated(train_dataset=dataset.tr_df, avg_rating_global=dataset.average_rating)
    hr_ndcg = hr.validate_ranking(val_dataset=dataset.sparse_te.toarray())
    logger.info(f"Highest Rated NDCG: {hr_ndcg}")
