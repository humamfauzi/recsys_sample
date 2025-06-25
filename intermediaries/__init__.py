"""
Shared definitions, enums, and classes for both train and serve modules.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict


class DataType(Enum):
    """Enum for different data types."""
    USER = "user"
    PRODUCT = "product"
    RATING = "rating"


class ModelType(Enum):
    """Enum for different model types."""
    ALS = "als"


@dataclass
class ModelConfig:
    """Configuration for model training and serving."""
    model_type: ModelType
    parameters: Dict[str, Any]


@dataclass
class UserData:
    """Shared user data structure."""
    user_id: Any
    metadata: Dict[str, Any]


@dataclass
class ProductData:
    """Shared product data structure."""
    product_id: Any
    metadata: Dict[str, Any]


@dataclass
class RatingData:
    """Shared rating data structure."""
    user_id: Any
    product_id: Any
    rating: float
