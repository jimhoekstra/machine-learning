import numpy as np
from .mse import mse


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))
