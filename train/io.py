"""
Data I/O module for reading and saving training data and models.
Contains interface and implementation classes for data operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any
from dataclasses import dataclass
from intermediaries.dataclass import BaseData, TrainingResult
import random
import string
import time
import os
import json


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
        product_file_path = self.data_path + "/item"
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
            return np.array(splitted).astype(np.float64)

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

    def generate_random_string(self, length: int = 10) -> str:
        """Generate a random string of fixed length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def save_training_result(self, training_result: TrainingResult) -> None:
        """Save training result to file."""
        randd = self.generate_random_string(2)
        current_time = time.strftime("%Y%m%d")
        train_result_path = self.data_path + f"/{current_time}_{randd}"
        os.makedirs(train_result_path, exist_ok=True)
        metadata = {
            'train_id': randd,
            'parameters': training_result.parameters,
            'user_index_map': training_result.user_index_map,
            'product_index_map': training_result.product_index_map,
            'global_mean': training_result.global_mean,
            'final_loss': training_result.final_loss
        }
        with open(train_result_path + "/metadata.json", 'w') as f:
            json.dump(metadata, f)

        np.save(train_result_path + "/user_weights.npy", training_result.user_weights)
        np.save(train_result_path + "/item_weights.npy", training_result.item_weights)
        np.save(train_result_path + "/user_bias.npy", training_result.user_bias)
        np.save(train_result_path + "/item_bias.npy", training_result.item_bias)
        np.save(train_result_path + "/user_metadata_weights.npy", training_result.user_metadata_weights)
        np.save(train_result_path + "/item_metadata_weights.npy", training_result.item_metadata_weights)
        return
