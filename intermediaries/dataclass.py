from dataclasses import dataclass
import numpy as np

@dataclass
class BaseData:
    """base data is the data transfer object for train"""
    rating: np.ndarray
    user: np.ndarray
    product: np.ndarray