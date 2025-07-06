"""
Training module containing ALS implementation and training pipeline.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import itertools
from intermediaries.dataclass import ProcessedTrainingData, ALSHyperParameters, TrainingResult, RangeIndex, Folds

class ALSModel:
    """Alternating Least Squares model implementation."""
    
    def __init__(self, n_iter: int, latent_factors: int, regularization: float, eta: float = 0.01):
        """Initialize ALS model with parameters."""
        self.n_iter = n_iter
        self.latent_factors = latent_factors
        self.regularization = regularization
        self.eta = eta  # Learning rate for metadata weights
        self.user_idx_map = {}
        self.product_idx_map = {}

    def initialize_weights_and_bias(self, n_users: int, n_items: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Initialize user and item weights and biases."""
        user_latent_weights = np.random.rand(n_users, self.latent_factors)
        prod_latent_weights = np.random.rand(n_items, self.latent_factors)

        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)
        return user_latent_weights, prod_latent_weights, user_bias, item_bias

    def initialize_metadata_weights(self, user_metadata_range: range, product_metadata_range: range) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Initialize metadata weights for users and products."""
        user_metadata_weights = np.random.rand(len(user_metadata_range))
        product_metadata_weights = np.random.rand(len(product_metadata_range))
        return user_metadata_weights, product_metadata_weights

    def generate_latent_factors(self,
        row: NDArray[np.float64],
        user_latent_weights: NDArray[np.float64],
        prod_latent_weights: NDArray[np.float64],
        user_metadata_weights: NDArray[np.float64],
        prod_metadata_weights: NDArray[np.float64],
        user_metadata_range: range = None,
        prod_metadata_range: range = None
    ) -> float:
        user, item = int(row[0]), int(row[1])
        user_idx = self.user_idx_map.get(user, -1)
        if user_idx == -1:
            raise ValueError(f"User {user} not found in user index map.")
        product_idx = self.product_idx_map.get(item, -1)
        if product_idx == -1:
            raise ValueError(f"Product {item} not found in product index map.")
        user_metadata = np.dot(row[user_metadata_range], user_metadata_weights)
        item_metadata = np.dot(row[prod_metadata_range], prod_metadata_weights)
        user_latent = user_latent_weights[user_idx] + user_metadata
        item_latent = prod_latent_weights[product_idx] + item_metadata
        latent = np.dot(user_latent, item_latent)
        return latent

    def find_unique_indices(self, data: NDArray[np.float64]) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
        """Find unique user and product indices."""
        unique_users = np.unique(data[:, 0]).astype(int)
        unique_products = np.unique(data[:, 1]).astype(int)
        
        user_index_map = {int(user): idx for idx, user in enumerate(unique_users)}
        product_index_map = {int(product): idx for idx, product in enumerate(unique_products)}
        return user_index_map, product_index_map, len(unique_users), len(unique_products)

    def fit(self, training_data: NDArray, user_metadata_range: range, product_metadata_range: range) -> TrainingResult:
        """Train the ALS model on training data."""
        self.user_idx_map, self.product_idx_map, n_users, n_items = self.find_unique_indices(training_data)
        # TODO: use enum to determine the column indices
        global_mean = np.mean(training_data[:, 2])  # Assuming ratings are in the third column
        user_latent_weights, prod_latent_weights, user_bias, item_bias = self.initialize_weights_and_bias(n_users, n_items)
        user_metadata_weights, prod_metadata_weights = self.initialize_metadata_weights(user_metadata_range, product_metadata_range)
        self.loss_iter_pair = []

        for iteration in range(self.n_iter):
            grad_user_metadata, grad_prod_metadata = 0, 0
            for row in training_data:
                # TODO: use enum to determine the column indices
                user_id, product_id, actual_rating = row[0], row[1], row[2]
                loss, residual = self.loss_function(
                    global_mean,
                    user_latent_weights,
                    prod_latent_weights,
                    user_bias,
                    item_bias,
                    user_metadata_weights,
                    prod_metadata_weights,
                    # TODO: use enum to determine the column indices
                    regularization=self.regularization,
                    row=row,
                )
                grad_user_metadata += residual * np.outer(prod_latent_weights[self.product_idx_map[product_id]], row[user_metadata_range])
                grad_prod_metadata += residual * np.outer(user_latent_weights[self.user_idx_map[user_id]], row[product_metadata_range])
            self.loss_iter_pair.append((iteration, loss))
            latent = self.generate_latent_factors(
                row, 
                user_latent_weights, 
                prod_latent_weights, 
                user_metadata_weights,
                prod_metadata_weights, 
                user_metadata_range, 
                product_metadata_range
            )
            # Update weights based on the loss
            # This is where the ALS update logic will go
            user_bias, item_bias = self.update_bias(
                self.user_idx_map[user_id],
                self.product_idx_map[product_id],
                loss, 
                global_mean,
                user_bias, 
                item_bias,
                training_data,
                latent
            )
            user_latent_weights, prod_latent_weights = self.update_latent_factors(
                user_id, 
                product_id, 
                training_data, 
                global_mean, 
                user_bias, 
                item_bias, 
                prod_latent_weights, 
                user_latent_weights
            )
            user_metadata_weights, prod_metadata_weights = self.update_metadata_weights(
                loss, 
                user_metadata_weights,
                prod_metadata_weights,
                grad_user_metadata,
                grad_prod_metadata,
                self.eta
            )
        return TrainingResult(
            parameters={
                'n_iter': self.n_iter,
                'latent_factors': self.latent_factors,
                'regularization': self.regularization,
                'eta': self.eta
            },
            user_weights=user_latent_weights,
            item_weights=prod_latent_weights,
            user_metadata_weights=user_metadata_weights,
            item_metadata_weights=prod_metadata_weights,
            user_bias=user_bias,
            item_bias=item_bias,
            user_index_map=self.user_idx_map,
            product_index_map=self.product_idx_map,
            global_mean=global_mean,
            final_loss=self.loss_iter_pair[-1][1]
        )

    def predict_order(self, training_result: TrainingResult, user_id: int, product_index_map: Dict[int, int]) -> List[int]:
        rank = []
        for product_key_id, product_id in product_index_map.items():
            result = self.predict(training_result, user_id, product_id)
            rank.append([product_key_id, result])
        rank.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in rank]

    def predict(self, training_result: TrainingResult, user_id: int, product_id: int) -> float:
        """Make a prediction for a given user and product."""
        user_latent = training_result.user_weights[user_id]
        item_latent = training_result.item_weights[product_id]
        user_b = training_result.user_bias[user_id]
        item_b = training_result.item_bias[product_id]
        global_mean = training_result.global_mean

        prediction = (user_latent @ item_latent) + user_b + item_b + global_mean
        return prediction

    def loss_function(self, 
        global_mean: float, 
        user_latent_weights: NDArray[np.float64], 
        prod_latent_weights: NDArray[np.float64],
        user_bias: NDArray[np.float64],
        item_bias: NDArray[np.float64],
        user_metadata_weights: NDArray[np.float64],
        prod_metadata_weights: NDArray[np.float64],
        regularization: float,
        row: NDArray[np.float64],
        latent: float = 0.0,
    ) -> float:
        """Compute the loss function."""
        # Implementation will go here
        user, item, actual_rating = int(row[0]), int(row[1]), row[2]  # Assuming user, item, rating are in the first three columns
        user_idx = self.user_idx_map.get(user, -1)
        if user_idx == -1:
            raise ValueError(f"User {user} not found in user index map.")

        product_idx = self.product_idx_map.get(item, -1)
        if product_idx == -1:
            raise ValueError(f"Product {item} not found in product index map.")

        # find user and item biases, if not found, return 0.0
        user_b = user_bias[user_idx]
        item_b = item_bias[product_idx]

        metadata_regularization = np.sum(user_metadata_weights ** 2) + np.sum(prod_metadata_weights ** 2)
        latent_regularization = np.sum(user_latent_weights[user_idx] ** 2) + np.sum(prod_latent_weights[product_idx] ** 2)
        bias_regularization = np.sum(user_bias ** 2) + np.sum(item_bias ** 2)
        regularization_term = regularization * (latent_regularization + bias_regularization + metadata_regularization)
        residual = (actual_rating - (latent + user_b + item_b + global_mean))
        final_loss = residual ** 2 + regularization_term

        return final_loss, residual

    def loss_function_residual(self,
        user_id: int,
        prod_id: int,
        actual_rating: float,
        global_mean: float, 
        user_bias: NDArray[np.float64],
        item_bias: NDArray[np.float64],
        latent: float = 0.0,
    ) -> float:
        """Update the residual loss"""
        user_b = user_bias[user_id]
        item_b = item_bias[prod_id]
        residual = (actual_rating - (latent + user_b + item_b + global_mean))
        return residual

    def update_latent_factors(self,
        user_id: int,
        product_id: int,
        training_data: NDArray[np.float64],
        global_mean: float,
        user_bias: NDArray[np.float64],
        item_bias: NDArray[np.float64],
        prod_latent_weights: NDArray[np.float64],
        user_latent_weights: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        rated_prod = np.where(training_data[:, 0] == user_id)[0]
        unique_products_rated_by_user = np.unique(training_data[rated_prod, 1]).astype(int)
        filtered_latent_prod = np.array([prod_latent_weights[self.product_idx_map[prod]] for prod in unique_products_rated_by_user])
        ratings_by_user = training_data[rated_prod, 2]
        
        residuals = []
        for i, prod in enumerate(unique_products_rated_by_user):
            prod_idx = self.product_idx_map[prod]
            residual = self.loss_function_residual(
                self.user_idx_map[user_id], 
                prod_idx, 
                ratings_by_user[i],
                global_mean, 
                user_bias, 
                item_bias,
                0.0  # latent placeholder
            )
            residuals.append(residual)
        
        new_user_latent = self.solve_latent(np.array(residuals), filtered_latent_prod, self.regularization)

        rated_users = np.where(training_data[:, 1] == product_id)[0]
        unique_users_rated_product = np.unique(training_data[rated_users, 0]).astype(int)
        filtered_latent_user = np.array([user_latent_weights[self.user_idx_map[user]] for user in unique_users_rated_product])
        ratings_for_product = training_data[rated_users, 2]
        
        residuals = []
        for i, user in enumerate(unique_users_rated_product):
            user_idx = self.user_idx_map[user]
            residual = self.loss_function_residual(
                user_idx,
                self.product_idx_map[product_id],
                ratings_for_product[i],
                global_mean,
                user_bias,
                item_bias,
                0.0  # latent placeholder
            )
            residuals.append(residual)
            
        new_prod_latent = self.solve_latent(np.array(residuals), filtered_latent_user, self.regularization)
        
        # Update the latent factors in the original arrays
        user_latent_weights[self.user_idx_map[user_id]] = new_user_latent
        prod_latent_weights[self.product_idx_map[product_id]] = new_prod_latent
        
        return user_latent_weights, prod_latent_weights

    def solve_latent(self,
        residuals: NDArray[np.float64],
        filtered: NDArray[np.float64],
        regularization: float,
    ) -> NDArray[np.float64]:
        try:
            A = np.dot(filtered.T, filtered) + regularization * np.eye(self.latent_factors)
            B = filtered.T @ np.array(residuals)
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            A = np.dot(filtered.T, filtered) + regularization * np.eye(self.latent_factors)
            B = filtered.T @ np.array(residuals)
            return np.linalg.pinv(A) @ B
        

    def update_bias(self, 
        user_id: int,
        item_id: int,
        loss: float, 
        global_mean: float,
        user_bias: NDArray[np.float64], 
        item_bias: NDArray[np.float64],
        training_data: NDArray[np.float64],
        latent: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update user and item biases."""
        # Find all items rated by this user
        user_indices = np.where(training_data[:, 0] == user_id)[0]
        # Find all users who rated this item
        item_indices = np.where(training_data[:, 1] == item_id)[0]
        
        updated_user_bias = np.sum(loss - global_mean - item_bias[item_id] - latent) 
        updated_user_bias = updated_user_bias / (len(user_indices) + self.regularization)

        updated_item_bias = np.sum(loss - global_mean - user_bias[user_id] - latent)
        updated_item_bias = updated_item_bias / (len(item_indices) + self.regularization)
        
        # Update the bias arrays
        user_bias[user_id] = updated_user_bias
        item_bias[item_id] = updated_item_bias
        
        return user_bias, item_bias

    def update_metadata_weights(self,
        loss: float,
        user_metadata_weights: NDArray[np.float64],
        prod_metadata_weights: NDArray[np.float64],
        grad_user_metadata: NDArray[np.float64],
        grad_prod_metadata: NDArray[np.float64],
        eta: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update user and product metadata weights."""

        # Ensure gradients are properly shaped
        if grad_user_metadata.ndim > 1:
            grad_user_metadata = grad_user_metadata.flatten()
        if grad_prod_metadata.ndim > 1:
            grad_prod_metadata = grad_prod_metadata.flatten()
            
        # Ensure gradients match weights dimensions
        if len(grad_user_metadata) != len(user_metadata_weights):
            grad_user_metadata = grad_user_metadata[:len(user_metadata_weights)]
        if len(grad_prod_metadata) != len(prod_metadata_weights):
            grad_prod_metadata = grad_prod_metadata[:len(prod_metadata_weights)]

        # Gradient descent update: w = w - eta * (regularization * w - gradient)
        step_user = eta * (grad_user_metadata - self.regularization * user_metadata_weights)
        new_user_metadata_weights = user_metadata_weights + step_user

        step_prod = eta * (grad_prod_metadata - self.regularization * prod_metadata_weights)
        new_prod_metadata_weights = prod_metadata_weights + step_prod

        return new_user_metadata_weights, new_prod_metadata_weights

class Trainer:
    """Main training class that manages the training process."""
    
    def __init__(self, hp: ALSHyperParameters):
        """Initialize the trainer with an ALS model."""
        self.hp = hp

    def get_data_from_indices(self, base_data, train_indices: List[RangeIndex], test_indices: RangeIndex):
        if not isinstance(train_indices, list):
            raise TypeError(f"Expected list of RangeIndex for train_indices, got {type(train_indices)}")
        if train_indices is None or len(train_indices) == 0:
            raise ValueError("train_indices must not be empty")
        tdata = np.concatenate([base_data[ti.start:ti.end] for ti in train_indices])
        vdata = base_data[test_indices.start:test_indices.end]
        return tdata, vdata
    
    def find_best_parameters(self, preprocessed_data: ProcessedTrainingData) -> TrainingResult:
        """Find best parameters using cross-validation."""
        possible_parameter = self.hp.to_dict()
        param_combinations = itertools.product(*possible_parameter.values())
        test_results = []
        training_results = []
        
        for pc in param_combinations:
            fold_indices_pick: Folds = np.random.choice(preprocessed_data.fold_indices, replace=False)
            params = dict(zip(possible_parameter.keys(), pc))
            model = ALSModel(**params)
            tdata, vdata = self.get_data_from_indices(preprocessed_data.training_data, fold_indices_pick.train_index, fold_indices_pick.test_index)
            tr = model.fit(tdata, preprocessed_data.user_metadata_range, preprocessed_data.product_metadata_range)
            training_results.append(tr)
            
            # Get a sample user for ranking
            user_id = int(vdata[0, 0])  # Use first user in validation set
            rank_list = model.predict_order(tr, user_id, tr.product_index_map)
            
            # Get actual ratings for this user
            user_ratings = vdata[vdata[:, 0] == user_id]
            sorted_ratings = user_ratings[user_ratings[:, 2].argsort()[::-1]]  # Sort by rating descending
            actual_order = sorted_ratings[:, 1].astype(int).tolist()  # Get item IDs in rating order
            
            # Calculate ranking distance
            dist = np.sum([0 if a == b else 1 for a, b in zip(rank_list, actual_order)])
            test_results.append([tr, dist])
            
        best_result = min(test_results, key=lambda x: x[1])
        return best_result[0]
