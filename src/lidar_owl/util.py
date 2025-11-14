import numpy as np
import torch

def tensor_to_np(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.array(value)