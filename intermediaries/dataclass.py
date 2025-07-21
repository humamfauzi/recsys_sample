from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional
from numpy.typing import NDArray

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
    train_index: List[RangeIndex]
    test_index: RangeIndex

@dataclass
class ProcessedTrainingData:
    """contains data that ready for train"""
    training_data: np.ndarray
    test_data: np.ndarray
    fold_indices: List[Folds]
    user_metadata_range: range
    product_metadata_range: range


@dataclass
class ALSHyperParameters:
    """Hyperparameters for ALS model."""
    n_iter: List[int]
    latent_factors: List[int]
    regularization: List[float]
    
    def to_dict(self) -> Dict:
        """Convert ALSHyperParameters to dictionary."""
        return {
            'n_iter': self.n_iter,
            'latent_factors': self.latent_factors,
            'regularization': self.regularization
        }

@dataclass
class TrainingResult:
    """Data class containing training results."""
    parameters: Dict[str, Any]
    user_weights: NDArray[np.float64]
    item_weights: NDArray[np.float64]
    user_metadata_weights: NDArray[np.float64]
    item_metadata_weights: NDArray[np.float64]
    user_bias: NDArray[np.float64]
    item_bias: NDArray[np.float64]
    user_index_map: Dict[int, int]
    product_index_map: Dict[int, int]
    global_mean: float
    final_loss: Optional[float]

def dummy_training_result() -> TrainingResult:
    """Create a dummy TrainingResult for testing purposes."""
    return TrainingResult(
        parameters={},
        user_weights=np.array([]),
        item_weights=np.array([]),
        user_metadata_weights=np.array([]),
        item_metadata_weights=np.array([]),
        user_bias=np.array([]),
        item_bias=np.array([]),
        user_index_map={},
        product_index_map={},
        global_mean=0.0,
        final_loss=None
    )   