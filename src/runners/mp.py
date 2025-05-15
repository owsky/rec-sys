from loguru import logger

from src.data_preprocessing.Dataset import Dataset
from src.models.non_personalized.MostPopular import MostPopular


def train_most_popular(dataset: Dataset):
    mp = MostPopular(train_dataset=dataset.tr_df)
    mp_ndcg = mp.validate_ranking(val_dataset=dataset.sparse_te.toarray())
    logger.info(f"Most Popular NDCG: {mp_ndcg}")
