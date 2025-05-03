import os

import dotenv

from src.data_preprocessing.load_dataset import load_dataset
from src.mf_als import tune_mf_als
from src.mf_als_mr import tune_mf_als_mr
from src.mf_sgd import tune_mf_sgd


def main():
    dotenv.load_dotenv()
    seed = 0
    dataset = load_dataset(seed=seed)

    model = os.getenv("MODEL")

    if model == "SGD":
        tune_mf_sgd(dataset=dataset)
    elif model == "ALS":
        tune_mf_als(dataset=dataset)
    elif model == "ALS_MR":
        tune_mf_als_mr(dataset=dataset)


if __name__ == "__main__":
    main()
