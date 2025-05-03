import numpy as np
from loguru import logger
from typing_extensions import override

from src.DataLoader import DataLoader
from src.models.Trainer import Trainer
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from src.utils.clip_gradient_norm import clip_gradient_norm


class MfSgdTrainer(Trainer):
    """
    Trainer class for Matrix factorization using stochastic gradient descent
    """

    model: MatrixFactorization

    def __init__(
        self, model: MatrixFactorization, tr_loader: DataLoader, reg: float, lr: float, early_patience=5
    ):
        """
        :param model: matrix factorization model to train
        :param tr_loader: data loader for training data
        :param reg: regularization parameter
        :param lr: learning rate
        :param early_patience: early stopping patience
        """
        self.model = model
        self.early_patience = early_patience
        self.reg = reg
        self.lr = lr
        self.tr_loader = tr_loader

    @override
    def training_epoch(self):
        """
        Training epoch for stochastic gradient descent
        """
        for users, items, ratings in self.tr_loader:
            # predict ratings for the batch using current parameters
            preds = self.model.predict(users, items)

            # if NaN values are present in preds, abort
            if not np.all(np.isfinite(preds)):
                logger.info("Early stopping...")
                return

            # compute the prediction errors
            errors_reshaped = np.square(preds - ratings)[:, np.newaxis]
            # compute the gradient updates
            grad_p = 2 * self.lr * (errors_reshaped * self.model.Q[items, :] - self.reg * self.model.P[users, :])
            grad_q = 2 * self.lr * (errors_reshaped * self.model.P[users, :] - self.reg * self.model.Q[items, :])
            # clip the gradients in case they grow too large
            grad_p = clip_gradient_norm(grad_p)
            grad_q = clip_gradient_norm(grad_q)
            # update the gradients for the batch
            self.model.P[users, :] += grad_p
            self.model.Q[items, :] += grad_q
