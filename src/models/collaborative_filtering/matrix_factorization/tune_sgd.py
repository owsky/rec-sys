import os
from typing import Optional, Any
import wandb
from src.DataLoader import DataLoader
from .MatrixFactorization import MatrixFactorization
from .SgdTrainer import SgdTrainer
from numpy.typing import NDArray


def tune_sgd(
    tune_config: dict[str, Any],
    tr: NDArray,
    val: NDArray,
    n_users: int,
    n_items: int,
    seed: int,
    entity_name: str,
    exp_name: str,
    sweep_name: str,
    bayesian_run_count: Optional[int] = 100,
):
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    def tune():
        with wandb.init():
            # get random configuration
            n_factors = wandb.config.get("n_factors")
            lr = wandb.config.get("learning_rate")
            reg = wandb.config.get("reg")
            batch_size = wandb.config.get("batch_size")
            tr_loader = DataLoader(data=tr, batch_size=batch_size, seed=seed)
            val_loader = DataLoader(data=val, batch_size=batch_size, seed=seed)
            mf = MatrixFactorization(n_factors=n_factors, n_users=n_users, n_items=n_items)
            trainer = SgdTrainer(model=mf)
            trainer.fit(
                tr_loader=tr_loader, val_loader=val_loader, n_epochs=1000, lr=lr, reg=reg, wandb_train=True
            )

    tune_config["name"] = sweep_name
    sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
    wandb.agent(sweep_id=sweep_id, function=tune, entity=entity_name, project=exp_name, count=bayesian_run_count)
