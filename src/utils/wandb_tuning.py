import os
from typing import Any, Optional, Callable

import wandb


def tune(
    tune_config: dict[str, Any],
    tune_fn: Callable,
    entity_name: str,
    exp_name: str,
    bayesian_run_count: Optional[int] = 100,
):
    """
    Wandb generic tuning of a model
    :param tune_config: tuning configuration dictionary
    :param tune_fn: tuning function to execute
    :param entity_name: Wandb entity name
    :param exp_name: Wandb project name
    :param bayesian_run_count: how many bayesian runs to execute
    """
    # log into Wandb using API key
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # create a new sweep
    sweep_id = wandb.sweep(sweep=tune_config, entity=entity_name, project=exp_name)
    # run the tuning
    wandb.agent(
        sweep_id=sweep_id, function=tune_fn, entity=entity_name, project=exp_name, count=bayesian_run_count
    )
