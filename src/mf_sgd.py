import os

from loguru import logger
from src.DataLoader import DataLoader
from src.data_preprocessing.Dataset import Dataset
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from src.models.collaborative_filtering.matrix_factorization.SgdTrainer import SgdTrainer
from src.models.collaborative_filtering.matrix_factorization.tune_sgd import tune_sgd


def train_mf_sgd(dataset: Dataset, seed: int):
    batch_size = 32
    n_factors = 100
    lr = 0.00000318119200608455
    reg = 0.0002070206242152593

    tr_loader = DataLoader(data=dataset.tr, batch_size=batch_size, seed=seed)
    val_loader = DataLoader(data=dataset.val, batch_size=batch_size, seed=seed)
    te_loader = DataLoader(data=dataset.te, batch_size=batch_size, seed=seed)

    mf = MatrixFactorization(n_factors=n_factors, n_users=dataset.n_users, n_items=dataset.n_items)
    sgd_trainer = SgdTrainer(model=mf)
    sgd_trainer.fit(tr_loader=tr_loader, val_loader=val_loader, lr=lr, reg=reg)

    test = sgd_trainer.validate(te_loader)
    logger.info(f"Final RMSE: {test}")


def tune_mf_sgd(dataset: Dataset, seed: int):
    tune_sgd(
        tune_config={
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "RMSE"},
            "parameters": {
                "n_factors": {"min": 1, "max": 100, "distribution": "int_uniform"},
                "learning_rate": {"min": 0.000001, "max": 0.01, "distribution": "log_uniform_values"},
                "reg": {"min": 0.00000001, "max": 0.01, "distribution": "log_uniform_values"},
                "batch_size": {"values": [32, 64, 128, 256, 512]},
            },
        },
        tr=dataset.tr,
        val=dataset.val,
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        seed=seed,
        sweep_name="MF_SGD",
        entity_name=os.environ.get("WANDB_ENTITY"),
        exp_name=os.environ.get("WANDB_PROJECT"),
    )
