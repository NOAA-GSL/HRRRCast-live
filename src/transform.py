import numpy as np

def log_transform_array(arr: np.ndarray) -> np.ndarray:
    """
    Apply log(1 + max(x, 0)) transform to a non-negative array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (values assumed >= 0, any negatives clipped to 0).
    eps : float, optional
        Small constant to stabilize the log (unused in this implementation).

    Returns
    -------
    np.ndarray
        Transformed array with log(1 + x) applied elementwise.
    """
    return np.log1p(np.clip(arr, 0, None))


def neg_log_transform_array(arr: np.ndarray) -> np.ndarray:
    """
    Apply signed log transform: sign(x) * log(1 + |x|).

    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain negative values.
    eps : float, optional
        Small constant to stabilize the log (unused in this implementation).

    Returns
    -------
    np.ndarray
        Transformed array with signed log(1 + |x|) applied elementwise.
    """
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_log_transform_array(arr: np.ndarray) -> np.ndarray:
    """
    Inverse of log_transform_array.

    For y = log(1 + x), returns x = exp(y) - 1.

    Parameters
    ----------
    arr : np.ndarray
        Transformed array.
    eps : float, optional
        Small constant to stabilize the log (unused in this implementation).

    Returns
    -------
    np.ndarray
        Inverse transformed array.
    """
    return np.expm1(arr)


def inverse_neg_log_transform_array(arr: np.ndarray) -> np.ndarray:
    """
    Inverse of neg_log_transform_array.

    For y = sign(x) * log(1 + |x|), returns x = sign(y) * (exp(|y|) - 1).

    Parameters
    ----------
    arr : np.ndarray
        Transformed array.
    eps : float, optional
        Small constant to stabilize the log (unused in this implementation).

    Returns
    -------
    np.ndarray
        Inverse transformed array.
    """
    return np.sign(arr) * np.expm1(np.abs(arr))
