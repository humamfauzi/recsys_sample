from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class BaseData:
    """base data is the data transfer object for train"""
    rating: np.ndarray
    user: np.ndarray
    product: np.ndarray

@dataclass
class RangeIndex:
    """contains range index for train and test"""
    start: int
    end: int

@dataclass
class Folds:
    TrainIndex: List[RangeIndex]
    TestIndex: RangeIndex

@dataclass
class ProcessedTrainingData:
    """contains data that ready for train"""
    training_data: np.ndarray
    test_data: np.ndarray
    fold_indices: List[Folds]