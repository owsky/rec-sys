import numpy as np
from numpy.typing import NDArray


def clip_gradient_norm(grad: NDArray, max_norm=1.0) -> NDArray:
    """
    Clip the given gradient to prevent explosion
    :param grad: numpy array containing the gradient to clip
    :param max_norm: maximum allowed norm of the gradient
    :return: the clipped gradient
    """
    # compute the norm of the gradient
    norm = np.linalg.norm(grad)
    # if the norm is greater than the maximum allowed norm
    if norm > max_norm:
        # scale it down
        grad = grad * (max_norm / float(norm))
    return grad
