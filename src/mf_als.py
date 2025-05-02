import os
import wandb
from loguru import logger
from src.DataLoader import DataLoader
from src.data_preprocessing.Dataset import Dataset
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from src.models.collaborative_filtering.matrix_factorization.AlsTrainer import AlsTrainer
from src.utils.wandb_tuning import tune


def train_mf_als(dataset: Dataset, seed: int):
    """
    Train the Matrix Factorization model using Alternating Least Squares
    :param dataset: dataset object
    :param seed: seed for reproducibility
    """
    # tuned hyper-parameters
    batch_size = 256
    n_factors = 1
    reg = 0.00991205549825312
    # create the data loaders
    val_loader = DataLoader(data=dataset.val, batch_size=batch_size, seed=seed)
    te_loader = DataLoader(data=dataset.te, batch_size=batch_size, seed=seed)
    # create the model
    mf = MatrixFactorization(n_factors=n_factors, n_users=dataset.n_users, n_items=dataset.n_items)
    # create the trainer
    als_trainer = AlsTrainer(model=mf, sparse_tr=dataset.sparse_tr)
    # train the model
    als_trainer.fit(val_loader=val_loader, reg=reg)
    # validate using test data loader
    test = als_trainer.validate(te_loader)
    logger.info(f"Final RMSE: {test}")


def tune_mf_als(dataset: Dataset, seed: int):
    """
    Tune the Matrix Factorization model using Alternating Least Squares
    :param dataset: dataset object
    :param seed: seed for reproducibility
    """
    # define tuning configuration
    tune_config = {
        "name": "MF_ALS",
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "RMSE"},
        "parameters": {
            "n_factors": {"min": 1, "max": 100, "distribution": "int_uniform"},
            "learning_rate": {"min": 0.000001, "max": 0.01, "distribution": "log_uniform_values"},
            "reg": {"min": 0.00000001, "max": 0.01, "distribution": "log_uniform_values"},
            "batch_size": {"values": [32, 64, 128, 256, 512]},
        },
    }

    # define the tuning function
    def tune_fn():
        with wandb.init():
            # get random configuration
            n_factors = wandb.config.get("n_factors")
            reg = wandb.config.get("reg")
            val_loader = DataLoader(data=dataset.val, batch_size=256, seed=seed)
            mf = MatrixFactorization(n_factors=n_factors, n_users=dataset.n_users, n_items=dataset.n_items)
            trainer = AlsTrainer(model=mf, sparse_tr=dataset.sparse_tr)
            trainer.fit(val_loader=val_loader, wandb_train=True, reg=reg)

    # tune the model
    tune(
        tune_config=tune_config,
        tune_fn=tune_fn,
        entity_name=os.getenv("WANDB_ENTITY"),
        exp_name=os.getenv("WANDB_PROJECT"),
    )
