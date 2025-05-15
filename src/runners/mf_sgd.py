import os
from typing import Optional

import wandb
from loguru import logger

from src.data_preprocessing.DataLoader import DataLoader
from src.data_preprocessing.Dataset import Dataset
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from src.models.collaborative_filtering.matrix_factorization.MfSgdTrainer import MfSgdTrainer
from src.utils.wandb_tuning import tune


def train_mf_sgd(dataset: Dataset, seed: Optional[int] = None):
    """
    Train the Matrix Factorization model using Stochastic Gradient Descent
    :param dataset: dataset object
    :param seed: seed for reproducibility
    """
    # tuned hyper-parameters
    batch_size = 32
    n_factors = 95
    lr = 0.00009083456271988014
    reg = 0.008300876421907001
    # create the data loaders
    tr_loader = DataLoader(data=dataset.tr, batch_size=batch_size, seed=seed)
    val_loader = DataLoader(data=dataset.val, batch_size=batch_size, seed=seed)
    te_loader = DataLoader(data=dataset.te, batch_size=batch_size, seed=seed)
    # create the model
    mf = MatrixFactorization(
        n_factors=n_factors,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        average_rating=dataset.average_rating,
        seed=seed,
    )
    # create the trainer
    sgd_trainer = MfSgdTrainer(model=mf, tr_loader=tr_loader, lr=lr, reg=reg)
    # train the model
    sgd_trainer.fit(val_loader=val_loader)
    # validate using test data loader
    rmse = mf.validate_prediction(te_loader)
    logger.info(f"MF SGD RMSE: {rmse}")
    ndcg = mf.validate_ranking(dataset.sparse_te.toarray())
    logger.info(f"MF SGD NDCG: {ndcg}")


def tune_mf_sgd(dataset: Dataset, seed: Optional[int] = None):
    """
    Tune the Matrix Factorization model using Stochastic Gradient Descent
    :param dataset: dataset object
    :param seed: seed for reproducibility
    """
    # define tuning configuration
    tune_config = {
        "name": "MF_SGD",
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
            lr = wandb.config.get("learning_rate")
            reg = wandb.config.get("reg")
            batch_size = wandb.config.get("batch_size")
            # create data loaders
            tr_loader = DataLoader(data=dataset.tr, batch_size=batch_size, seed=seed)
            val_loader = DataLoader(data=dataset.val, batch_size=batch_size, seed=seed)
            # create the model
            mf = MatrixFactorization(
                n_factors=n_factors,
                n_users=dataset.n_users,
                n_items=dataset.n_items,
                average_rating=dataset.average_rating,
                seed=seed,
            )
            # create the trainer
            trainer = MfSgdTrainer(model=mf, tr_loader=tr_loader, lr=lr, reg=reg)
            # train the model
            trainer.fit(val_loader=val_loader, wandb_train=True)

    # tune the model
    tune(
        tune_config=tune_config,
        tune_fn=tune_fn,
        entity_name=os.getenv("WANDB_ENTITY"),
        exp_name=os.getenv("WANDB_PROJECT"),
    )
