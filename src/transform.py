import numpy as np
import xarray as xr

# Reusable epsilon constant (can be imported elsewhere if needed)
DEFAULT_LOG_EPS = 1e-3


def log_transform_array(arr: np.ndarray, eps: float = DEFAULT_LOG_EPS) -> np.ndarray:
    """Apply log(x+eps)-log(eps) transform to a non-negative array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (values assumed >= 0, any negatives clipped to 0).
    eps : float
        Small constant to stabilize the log.
    """
    return np.log(np.clip(arr, 0, None) + eps) - np.log(eps)


def neg_log_transform_array(arr: np.ndarray, eps: float = DEFAULT_LOG_EPS) -> np.ndarray:
    """Apply signed log transform: sign(x)*(log(|x|+eps)-log(eps)).

    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain negative values.
    eps : float
        Small constant to stabilize the log.
    """
    return np.sign(arr) * (np.log(np.abs(arr) + eps) - np.log(eps))


def inverse_log_transform_array(arr: np.ndarray, eps: float = DEFAULT_LOG_EPS) -> np.ndarray:
    """Inverse of log_transform_array.

    y = log(x+eps) - log(eps)  =>  x = exp(y + log(eps)) - eps
    """
    return np.exp(arr + np.log(eps)) - eps


def inverse_neg_log_transform_array(arr: np.ndarray, eps: float = DEFAULT_LOG_EPS) -> np.ndarray:
    """Inverse of neg_log_transform_array.

    y = sign(x)*(log(|x|+eps)-log(eps))
    sign(x)=sign(y); |x| = exp(|y|+log(eps)) - eps
    """
    sign_x = np.sign(arr)
    abs_x = np.exp(np.abs(arr) + np.log(eps)) - eps
    return sign_x * abs_x