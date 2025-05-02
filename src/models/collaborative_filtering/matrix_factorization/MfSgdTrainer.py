import numpy as np
from src.DataLoader import DataLoader
from src.utils.clip_gradient_norm import clip_gradient_norm
from src.models.collaborative_filtering.matrix_factorization.MatrixFactorization import MatrixFactorization
from loguru import logger
from src.models.Trainer import Trainer


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

        # compute the global bias as average of all ratings
        self.model.global_bias = np.mean([np.mean(ratings) for _, _, ratings in tr_loader], dtype=np.float64)

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
            errors = np.square(preds - ratings)

            # reshape the errors to accomodate broadcasting
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
            # update the bias terms
            self.model.users_bias[users] += self.lr * (errors - self.reg * self.model.users_bias[users])
            self.model.items_bias[items] += self.lr * (errors - self.reg * self.model.items_bias[items])
