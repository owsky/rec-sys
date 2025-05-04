import copy
from abc import abstractmethod, ABC

import numpy as np
import wandb
from loguru import logger
from tqdm import tqdm

from src import DataLoader
from .Model import Model


class Trainer(ABC):
    """
    Abstract class which represents a generic trainer for a model
    """

    model: Model
    early_patience: int

    @abstractmethod
    def training_epoch(self, *args, **kwargs):
        pass

    def after_training(self, *args, **kwargs):
        """
        Method which is run after the training loop is over for cleanup
        """
        pass

    def fit(self, val_loader: DataLoader, n_epochs=1000, wandb_train=False, *args, **kwargs):
        """
        Run the training loop for the model
        :param val_loader: validation data loader
        :param n_epochs: number of epochs to train for
        :param wandb_train: whether the run should be logged using wandb
        """
        best_val_score = np.inf
        best_model = copy.deepcopy(self.model)
        curr_patience = 0

        for _ in tqdm(range(n_epochs), desc="Training the model...", dynamic_ncols=True):
            # run the training epoch defined by the concrete model
            self.training_epoch(*args, **kwargs)
            # compute the current validation score
            val_score = self.model.validate(val_loader=val_loader)

            # if wandb logging is enabled, log the current validation RMSE
            if wandb_train:
                wandb.log({"RMSE": val_score})

            # if the current validation score is better than the previously stored one
            if val_score < best_val_score:
                # update it
                best_val_score = val_score
                # if wandb logging is enabled, log the current best RMSE
                if wandb_train:
                    wandb.log({"Best RMSE": val_score})
                # store the current model object for early stopping
                best_model = copy.deepcopy(self.model)
                # reset early stopping counter
                curr_patience = 0
            else:
                # increase early stopping counter
                curr_patience += 1
                # if early stopping patience has been met
                if curr_patience == self.early_patience:
                    logger.info("Early stopping")
                    # terminate training and reload previously best performing model
                    self.model = best_model
                    break
        self.after_training()
