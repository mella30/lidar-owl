import numpy as np
import torch


def tensor_to_np(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.array(value)


def safe_divide(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(num, den)
    result = np.asarray(result, dtype=np.float64)
    if result.ndim == 0:
        return float(result) if den != 0 else float("nan")
    return np.where(den == 0, np.nan, result)
