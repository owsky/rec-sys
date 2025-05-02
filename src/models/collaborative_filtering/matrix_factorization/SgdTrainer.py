import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from src.DataLoader import DataLoader
from .MatrixFactorization import MatrixFactorization
from loguru import logger
from ...Trainer import Trainer
import wandb


def clip_gradient_norm(grad: NDArray, max_norm=1.0) -> NDArray:
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


def mean_squared_error(predictions: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> np.floating:
    difference = np.subtract(predictions, ground_truth)
    squared_difference = np.square(difference)
    mse = np.mean(squared_difference)
    return mse


def root_mean_squared_error(predictions: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> np.floating:
    mse = mean_squared_error(predictions, ground_truth)
    rmse = np.sqrt(mse)
    return rmse


class SgdTrainer(Trainer):
    def __init__(self, model: MatrixFactorization, early_patience=5):
        self.model = model
        self.early_patience = early_patience

    def fit(
        self,
        tr_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        reg: float,
        n_epochs=1000,
        wandb_train=False,
    ):
        best_val_score = np.inf
        best_p = self.model.P
        best_q = self.model.Q

        curr_patience = 0

        # compute the global bias
        means = []
        for _, _, ratings in tr_loader:
            means.append(np.mean(ratings))
        self.model.global_bias = np.mean(means, dtype=np.float64)

        for _ in tqdm(range(n_epochs), desc="Training the model...", dynamic_ncols=True):
            for users, items, ratings in tr_loader:
                preds = self.model.predict(users, items)

                if not np.all(np.isfinite(preds)):
                    logger.info("Early stopping...")
                    return

                errors = np.square(preds - ratings)[:, np.newaxis]
                grad_p = 2 * lr * (errors * self.model.Q[items, :] - reg * self.model.P[users, :])
                grad_q = 2 * lr * (errors * self.model.P[users, :] - reg * self.model.Q[items, :])

                grad_p = clip_gradient_norm(grad_p)
                grad_q = clip_gradient_norm(grad_q)

                self.model.P[users, :] += grad_p
                self.model.Q[items, :] += grad_q

                errors = np.square(preds - ratings)
                self.model.users_bias[users] += lr * (errors - reg * self.model.users_bias[users])
                self.model.items_bias[items] += lr * (errors - reg * self.model.items_bias[items])

            val_score = self.validate(val_loader=val_loader)

            if wandb_train:
                wandb.log({"RMSE": val_score})

            if val_score < best_val_score:
                best_val_score = val_score
                if wandb_train:
                    wandb.log({"Best RMSE": val_score})
                best_p = self.model.P
                best_q = self.model.Q
                curr_patience = 0
            else:
                curr_patience += 1
                if curr_patience == self.early_patience:
                    logger.info("Early stopping")
                    self.model.P = best_p
                    self.model.Q = best_q
                    break

    def validate(self, val_loader: DataLoader) -> np.float64:
        validation_values = []
        for users, items, ratings in val_loader:
            preds = self.model.predict(users, items)
            rmse = root_mean_squared_error(preds, ratings)
            validation_values.append(rmse)
        return np.mean(validation_values, dtype=np.float64)
