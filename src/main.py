import os

from src.DataLoader import DataLoader
from src.data_preprocessing.load_dataset import load_dataset
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from src.models.collaborative_filtering.matrix_factorization.SgdTrainer import SgdTrainer
from loguru import logger
import dotenv
from src.models.collaborative_filtering.matrix_factorization.tune_sgd import tune_sgd


def main():
    dotenv.load_dotenv()
    seed = 0
    dataset = load_dataset(seed=seed)


if __name__ == "__main__":
    main()
