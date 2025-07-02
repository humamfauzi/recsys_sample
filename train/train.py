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
        user, item = row[0], row[1]
        user_metadata = np.dot(row[user_metadata_range], user_metadata_weights)
        item_metadata = np.dot(row[prod_metadata_range], prod_metadata_weights)
        print("user", user)
        user_latent = user_latent_weights[int(user)] + user_metadata
        item_latent = prod_latent_weights[int(item)] + item_metadata
        latent = np.dot(user_latent, item_latent)
        return latent

    def find_unique_indices(self, data: NDArray[np.float64]) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
        """Find unique user and product indices."""
        unique_users = np.unique(data[:, 0])
        unique_products = np.unique(data[:, 1])
        
        user_index_map = {user: idx for idx, user in enumerate(unique_users)}
        product_index_map = {product: idx for idx, product in enumerate(unique_products)}
        return user_index_map, product_index_map, len(unique_users), len(unique_products)

    def fit(self, training_data: NDArray, user_metadata_range: range, product_metadata_range: range) -> TrainingResult:
        """Train the ALS model on training data."""
        self.user_idx_map, self.product_index_map, n_users, n_items = self.find_unique_indices(training_data)
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
                    user_metadata_range=user_metadata_range,
                    prod_metadata_range=prod_metadata_range
                )
                filtered_prod = np.where(training_data[:, 0] == user_id)
                grad_user_metadata += residual * np.dot(training_data[filtered_prod, user_metadata_range], user_metadata_weights)
                grad_prod_metadata += residual * np.dot(training_data[filtered_prod, product_metadata_range], prod_metadata_weights)
            self.loss_iter_pair.append((iteration, loss))
            latent = self.generate_latent_factors(
                row, 
                user_latent_weights, 
                prod_latent_weights, 
                user_metadata_weights,
                prod_metadata_weights, 
                user_metadata_range, 
                user_metadata_range
            )
            # Update weights based on the loss
            # This is where the ALS update logic will go
            user_bias, item_bias = self.update_bias(
                loss, 
                global_mean,
                user_latent_weights,
                prod_latent_weights,
                user_bias, 
                item_bias,
                user_metadata_weights,
                prod_metadata_weights,
                regularization=self.regularization,
                row=row,
                latent=latent
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
            user_bias=user_bias,
            item_bias=item_bias,
            user_index_map=self.user_idx_map,
            product_index_map=self.product_index_map,
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

        # find user and item biases, if not found, return 0.0
        user_b = user_bias[user]
        item_b = item_bias[item]

        metadata_regularization = np.sum(user_metadata_weights ** 2) + np.sum(prod_metadata_weights ** 2)
        latent_regularization = np.sum(user_latent_weights[user] ** 2) + np.sum(prod_latent_weights[item] ** 2) 
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
        rated_prod = np.where(training_data[:, 0] == user_id)
        unique_products_rated_by_user = np.unique(training_data[rated_prod, 1])
        filtered_latent_prod = np.array([prod_latent_weights[prod] for prod in unique_products_rated_by_user])
        residuals = [self.loss_function_residual(
            user_id, 
            prod, 
            training_data[rated_prod, 2],  # TODO: use enum to determine the column index for rating
            global_mean, 
            user_bias, 
            item_bias,
            filtered_latent_prod,
        ) for prod in unique_products_rated_by_user]
        new_user_latent = self.solve_latent(residuals, filtered_latent_prod, self.regularization)

        rated_users = np.where(training_data[:, 1] == product_id)
        unique_users_rated_product = np.unique(training_data[rated_users, 0])
        filtered_latent_user = np.array([user_latent_weights[user] for user in unique_users_rated_product])
        residuals = [self.loss_function_residual(
            user,
            product_id,
            training_data[rated_users, 2],  # TODO: use enum to determine the column index for rating
            global_mean,
            user_bias,
            item_bias,
            filtered_latent_user,
        ) for user in unique_users_rated_product]
        new_prod_latent = self.solve_latent(residuals, filtered_latent_user, self.regularization)
        return new_user_latent, new_prod_latent

    def solve_latent(self,
        residuals: NDArray[np.float64],
        filtered: NDArray[np.float64],
        regularization: float,
    ) -> NDArray[np.float64]:
        A = np.dot(filtered.T, filtered) + regularization * np.eye(self.latent_factors)
        B = filtered.T @ np.array(residuals)
        return np.linalg.solve(A, B)
        

    def update_bias(self, 
        item_id: int,
        user_id: int,
        loss: float, 
        global_mean: float,
        user_bias: NDArray[np.float64], 
        item_bias: NDArray[np.float64],
        training_data: NDArray[np.float64],
        latent: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update user and item biases."""
        user_indices = np.where(training_data[:, 0] == user_id)
        item_indices = np.where(training_data[:, 1] == item_id)

        updated_user_bias = np.sum(loss - global_mean - item_bias[item_indices] - latent) 
        updated_user_bias = updated_user_bias / (len(user_indices) + self.regularization)

        updated_item_bias = np.sum(loss - global_mean - user_bias[user_indices] - latent)
        updated_item_bias = updated_item_bias / (len(item_indices) + self.regularization)
        return updated_user_bias, updated_item_bias

    def update_metadata_weights(self,
        user_metadata_weights: NDArray[np.float64],
        prod_metadata_weights: NDArray[np.float64],
        gradient: float,
        eta: float = 0.01
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update user and product metadata weights."""

        step = 2 * eta * (gradient - self.regularization * user_metadata_weights)
        new_user_metadata_weights = user_metadata_weights +  step

        step = 2 * eta * (gradient - self.regularization * prod_metadata_weights)
        new_prod_metadata_weights = prod_metadata_weights + step

        return new_user_metadata_weights, new_prod_metadata_weights

class Trainer:
    """Main training class that manages the training process."""
    
    def __init__(self, hp: ALSHyperParameters):
        """Initialize the trainer with an ALS model."""
        self.hp = hp

    def get_data_from_indices(self, base_data, train_indices: List[RangeIndex], test_indices: RangeIndex):
        if train_indices:
            tdata = np.concatenate([base_data[ti.start:ti.end] for ti in train_indices])
        else:
            # Handle empty train_indices case
            tdata = np.empty((0, base_data.shape[1]), dtype=base_data.dtype)
        vdata = base_data[test_indices.start:test_indices.end]
        return tdata, vdata
    
    def find_best_parameters(self, preprocessed_data: ProcessedTrainingData) -> TrainingResult:
        """Find best parameters using cross-validation."""
        possible_parameter = self.hp.to_dict()
        param_combinations = itertools.product(*possible_parameter.values())
        test_results = []
        for pc in param_combinations:
            fold_indices_pick = np.random.choice(preprocessed_data.fold_indices, size=len(preprocessed_data.fold_indices), replace=False)
            fold = preprocessed_data.fold_indices[fold_indices_pick]
            params = dict(zip(possible_parameter.keys(), pc))
            model = ALSModel(**params)
            tdata, vdata = self.get_data_from_indices(preprocessed_data.training_data, fold.train_index, fold.test_index)
            tr = model.fit(tdata, preprocessed_data.user_metadata_range, preprocessed_data.product_metadata_range)
            training_results.append(tr)
            rank_list = model.predict_order(vdata, user_id, tr.product_index_map)
            sliced = np.where(vdata, vdata[:, 0] == user_id)
            sorted(sliced, lambda x: x[3], reverse=True)
            dist = np.sum([0 if a == b else 1 for a, b in zip(rank_list, sliced)])
            test_results.append([tr, dist])
        best_result = min(test_results, key=lambda x: x[1])
        return best_result
