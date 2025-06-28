"""
Data I/O module for reading and saving training data and models.
Contains interface and implementation classes for data operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any
from dataclasses import dataclass
from intermediaries.dataclass import BaseData


class DataIOInterface(ABC):
    """Interface for data I/O operations."""
    @abstractmethod
    def read_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read all data (users, products, ratings) and return as a tuple of numpy arrays."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def save_model(self, model_data: Any) -> None:
        """Save model data to file."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class DataIO(DataIOInterface):
    """Implementation of data I/O operations."""
    
    def __init__(self, data_path: str = "dataset"):
        """Initialize with data path."""
        self.data_path = data_path
    
    def read_users(self) -> np.ndarray:
        """Read users data and return as numpy array."""
        user_file_path = self.data_path + "/user"
        with open(user_file_path, 'r') as file:
            data = file.read().strip().split('\n')
            splitted = [line.split('|') for line in data]
            return np.array(splitted)
    
    def read_products(self) -> np.ndarray:
        """Read products data and return as numpy array."""
        product_file_path = self.data_path + "/product"
        with open(product_file_path, 'r') as file:
            data = file.read().strip().split('\n')
            splitted = [line.split('|') for line in data]
            return np.array(splitted)
    
    def read_ratings(self) -> np.ndarray:
        """Read ratings data and return as numpy array."""
        ratings_file_path = self.data_path + "/rating"
        with open(ratings_file_path, 'r') as file:
            data = file.read().strip().split('\n')
            splitted = [line.split('\t') for line in data]
            return np.array(splitted)

    def read_all(self) -> BaseData:
        """Read all data (users, products, ratings) and return as a BaseData instance."""
        users = self.read_users()
        products = self.read_products()
        ratings = self.read_ratings()
        return BaseData(rating=ratings, user=users, product=products)
    
    def save_model(self, model_data: np.ndarray) -> None:
        """Save model data to file."""
        np.save(self.data_path + "/model.npy", model_data)
        return
