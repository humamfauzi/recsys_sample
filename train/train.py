"""
Training module containing ALS implementation and training pipeline.
"""

import numpy as np
import time
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
        user_latent_weights = np.random.normal(0, 0.1, (n_users, self.latent_factors))
        prod_latent_weights = np.random.normal(0, 0.1, (n_items, self.latent_factors))

        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)
        return user_latent_weights, prod_latent_weights, user_bias, item_bias

    def initialize_metadata_weights(self, n_user: int, n_item: int, user_metadata_range: range, product_metadata_range: range) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Initialize metadata weights for users and products."""
        user_metadata_weights = np.random.normal(0, 0.1, (n_user, self.latent_factors, len(user_metadata_range)))
        product_metadata_weights = np.random.normal(0, 0.1, (n_item, self.latent_factors, len(product_metadata_range)))
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

    def fit(self, training_data: NDArray[float], user_metadata_range: range, product_metadata_range: range) -> TrainingResult:
        """Train the ALS model on training data."""
        self.user_idx_map, self.product_idx_map, n_users, n_items = self.find_unique_indices(training_data)
        # TODO: use enum to determine the column indices
        global_mean = np.mean(training_data[:, 2])  # Assuming ratings are in the third column
        user_latent_weights, item_latent_weights, user_bias, item_bias = self.initialize_weights_and_bias(n_users, n_items)
        user_metadata_weights, item_metadata_weights = self.initialize_metadata_weights(n_users, n_items, user_metadata_range, product_metadata_range)
        self.loss_iter_pair = []

        # total loops for fitting calculation should be number of iteration times number of training data rows
        for iteration in range(self.n_iter):
            start = time.time()
            buffer_user_latent_weights, buffer_item_latent_weights, buffer_user_bias, buffer_item_bias = self.initialize_weights_and_bias(n_users, n_items)
            # Step 1: Update the latent and bias for users; making the product fixed
            # this should filter all training data to highlighted user
            for user_id, user_idx in self.user_idx_map.items():
                # ids in training data rows are not translated yet, in order to access 
                # user's or item's weights, we need to translate it with a map
                user_data = self.select_intersect(training_data, user_id, 0)

                # all items that this user has interacted with
                unique_items = np.unique(user_data[:, 1]).astype(int)

                # get the indices of unique items in the product index map so we can access the bias and latent weights of this particular items
                unique_items_idx = [self.product_idx_map[item_id] for item_id in unique_items]

                # calculate the latent matrix for items that this user has interacted with
                # based on equation $A = \sum_{j \in \Omega_i} (V_j + W_g G_j) * (V_j + W_g G_j)^T + \lambda I$
                # we can use the matrix multiplication to calculate the latent matrix for items
                # $A = (V_\omega_i + W_g G_\omega_i) * (V_\omega_i + W_g G_\omega_i)^T + \lambda I$
                # where $V_\omega_i$ (n_interacted_item, n_latent_factor) is the latent weights for items that user $i interacted with
                # where $W_g$ (n_interacted_item, n_metadata_item, n_latent_factor) is the metadata weights for items that user $i$ interacted with
                # where $G_\omega_i$ (n_interacted_item, n_metadata_item) is the metadata value for items that user $i$ interacted with
                # use einstein summation to calculate the dot product of metadata weights and metadata values; where k represented the metadata feature index
                latent_metadata = item_latent_weights[unique_items_idx] + np.einsum("ijk,ik->ij", item_metadata_weights[unique_items_idx], user_data[:, product_metadata_range])
                # smaller interaction will have larger regularization, this is to prevent overfitting and ill conditioned matrix
                adaptive_regularization = self.regularization / len(unique_items) ** 0.5
                latent_matrix_items = np.matmul(latent_metadata.T, latent_metadata) + adaptive_regularization * np.eye(self.latent_factors)
                # accumulate the residual based
                # this is the equation $B = \sum_{i \in i_u} (r_{ui} - \mu - b_i - b_u - m_u^t p_i) p_i$

                # bias accumulator based on 
                # this is the equation $b_i = \sum_{j \in \Omega_i} (R_{ij} - \mu - b_j - (U_i + W_a A_i) \cdot V_j)$
                vector_acc = np.zeros((self.latent_factors,))
                bias_acc = 0
                for row in user_data:
                    item_idx = self.product_idx_map[int(row[1])]
                    # $(V_j + W_g G_j)$ fixing item when calculating user; should have shape of (n_latent_factors,)
                    item_fixed = item_latent_weights[item_idx]+ np.matmul(item_metadata_weights[item_idx], row[product_metadata_range])
                    # $ (W_a A_i) \cdot (V_j + W_g G_j) $ should have scalar value
                    # $W_a$ is the metadata weights for item with shape (n_metadata_user, n_latent_factors)
                    # $A_i$ is the metadata value for user with index $i$ with shape (n_metadata_user,)
                    # $V_j$ is the latent weights for item with index $j$ with shape (n_latent_factors,)
                    # $W_g$ is the metadata weights for user with shape (n_metadata_item, n_latent_factors)
                    # $G_j$ is the metadata value for item with index $j$ with shape (n_metadata_item,)
                    mm = np.dot(np.matmul(user_metadata_weights[user_idx], row[user_metadata_range]), item_fixed)
                    residual = row[2] - global_mean - item_bias[item_idx] - user_bias[user_idx] - mm
                    vector_acc += residual * item_fixed
                    item_latent_and_metadata = item_latent_weights[item_idx] + np.matmul(item_metadata_weights[item_idx], row[product_metadata_range])
                    user_latent_and_metadata = user_latent_weights[user_idx] + np.matmul(user_metadata_weights[user_idx], row[user_metadata_range])
                    bias_acc += row[2] - global_mean - item_bias[item_idx] - np.dot(item_latent_and_metadata, user_latent_and_metadata)
                # solve the linear equation to get the new user latent weights
                # this is the equation $A_u x_u = B_u$
                # the shape should be (1, latent_factors) = (latent_factors, latent_factors) * (1, latent_factors)
                solve = np.linalg.solve(latent_matrix_items, vector_acc)
                buffer_user_latent_weights[user_idx] = solve / (np.linalg.norm(solve) + self.regularization)
                if np.mean(buffer_user_latent_weights[user_idx]) > 10:
                    print(f"user_idx: {user_idx}, solution {buffer_user_latent_weights[user_idx]}, matrix: {latent_matrix_items} vector_acc: {vector_acc}")
                buffer_user_bias[user_idx] = bias_acc / (len(unique_items) + self.regularization)

            # Step 2: Update the latent and bias for items; making the user fixed
            # this would take the whole training set 
            for item_id, item_idx in self.product_idx_map.items():
                item_data = self.select_intersect(training_data, item_id, 1)
                unique_users = np.unique(item_data[:, 0]).astype(int)
                unique_users_idx = [self.user_idx_map[user_id] for user_id in unique_users]
                latent_metadata = user_latent_weights[unique_users_idx] + np.einsum("ijk,ik->ij", user_metadata_weights[unique_users_idx], item_data[:, user_metadata_range])
                adaptive_regularization = self.regularization / len(unique_users) ** 0.5
                latent_matrix_users = np.matmul(latent_metadata.T, latent_metadata) + adaptive_regularization * np.eye(self.latent_factors)

                bias_acc = 0
                vector_acc = np.zeros((self.latent_factors,))

                for row in item_data:
                    user_idx = self.user_idx_map[int(row[0])]
                    user_fixed = user_latent_weights[user_idx] + np.matmul(user_metadata_weights[user_idx], row[user_metadata_range])
                    mm = np.dot(np.matmul(item_metadata_weights[item_idx], row[product_metadata_range]), user_fixed)
                    residual = row[2] - global_mean - item_bias[item_idx] - user_bias[user_idx] - mm
                    vector_acc += residual * user_fixed
                    item_latent_and_metadata = item_latent_weights[item_idx] + np.matmul(item_metadata_weights[item_idx], row[product_metadata_range])
                    user_latent_and_metadata = user_latent_weights[user_idx] + np.matmul(user_metadata_weights[user_idx], row[user_metadata_range])
                    bias_acc += row[2] - global_mean - user_bias[user_idx] - np.dot(item_latent_and_metadata, user_latent_and_metadata)

                solve = np.linalg.solve(latent_matrix_users, vector_acc)
                buffer_item_latent_weights[item_idx] = solve / (np.linalg.norm(solve) + self.regularization)
                if np.mean(buffer_item_latent_weights[item_idx]) > 10:
                    print(f"item idx: {item_idx}")
                    print(f"new_latent: {buffer_item_latent_weights[item_idx]}, prev_latent: {item_latent_weights[item_idx]}")
                    print(f"matrix: {latent_matrix_users}")
                    print(f"latent_metadata: {latent_metadata}")
                    print(f"vector: {vector_acc}")
                    eigenvalues = np.linalg.eigvals(latent_matrix_users)
                    print(f"latent_matrix_users eigenvalues: {np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))}")
                buffer_item_bias[item_idx] = bias_acc / (len(unique_users) + self.regularization)

            # Step 3: Update metadata using accumulated gradients
            user_acc, item_acc = np.zeros_like(user_metadata_weights), np.zeros_like(item_metadata_weights)
            for row in training_data:
                user_idx, product_idx = self.user_idx_map[int(row[0])], self.product_idx_map[int(row[1])]
                rating = row[2]
                ilatent = item_latent_weights[product_idx] + np.dot(item_metadata_weights[product_idx], row[product_metadata_range])
                ulatent = user_latent_weights[user_idx] + np.dot(user_metadata_weights[user_idx], row[user_metadata_range])
                residual = rating - (global_mean + user_bias[user_idx] + item_bias[product_idx] + np.dot(ulatent, ilatent))
                user_acc[user_idx] += (residual * ilatent).reshape(self.latent_factors, -1) * \
                    np.array(row[user_metadata_range]).reshape(-1, len(user_metadata_range)) - \
                    self.regularization * user_metadata_weights[user_idx]
                item_acc[product_idx] += (residual * ulatent).reshape(self.latent_factors, -1) * \
                    np.array(row[product_metadata_range]).reshape(-1, len(product_metadata_range)) - \
                    self.regularization * item_metadata_weights[product_idx]

            user_metadata_weights += self.eta * user_acc / len(training_data)
            item_metadata_weights += self.eta * item_acc / len(training_data)

            user_latent_weights = np.copy(buffer_user_latent_weights)
            item_latent_weights = np.copy(buffer_item_latent_weights)
            user_bias = np.copy(buffer_user_bias)
            item_bias = np.copy(buffer_item_bias)


            if iteration % 1 == 0:
                tr = TrainingResult(
                    parameters={
                        'n_iter': self.n_iter,
                        'latent_factors': self.latent_factors,
                        'regularization': self.regularization,
                        'eta': self.eta
                    },
                    user_weights=user_latent_weights,
                    item_weights=item_latent_weights,
                    user_metadata_weights=user_metadata_weights,
                    item_metadata_weights=item_metadata_weights,
                    user_bias=user_bias,
                    item_bias=item_bias,
                    user_index_map=self.user_idx_map,
                    product_index_map=self.product_idx_map,
                    global_mean=global_mean,
                    final_loss=None
                )
                current_loss, diff, reg = self.calculate_loss(tr, training_data, user_metadata_range, product_metadata_range)  
                print(f"Iteration: {iteration}, Loss: {current_loss}, Diff: {diff}, Reg: {reg}, Time: {time.time() - start:.2f}s")
                self.loss_iter_pair.append((iteration, current_loss))
                tr.final_loss = current_loss
        return tr

    def predict_order(self, training_result: TrainingResult, user_id: int, product_index_map: Dict[int, int]) -> List[int]:
        rank = []
        for product_key_id, product_id in product_index_map.items():
            result = self.predict(training_result, user_id, product_id)
            rank.append([product_key_id, result])
        rank.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in rank]

    def calculate_loss(self, training_result: TrainingResult, training_data: NDArray[np.float64], user_metadata_range: range, product_metadata_range: range) -> float:
        """Calculate the loss for the training data."""
        total_loss = 0.0
        for row in training_data:
            user_id, product_id, actual_rating =  self.user_idx_map[int(row[0])], self.product_idx_map[int(row[1])], row[2]
            user_latent = training_result.user_weights[user_id] + np.dot(training_result.user_metadata_weights[user_id], row[user_metadata_range])
            item_latent = training_result.item_weights[product_id] + np.dot(training_result.item_metadata_weights[product_id], row[product_metadata_range])
            user_b = training_result.user_bias[user_id]
            item_b = training_result.item_bias[product_id]
            global_mean = training_result.global_mean
            latent_dot = user_latent @ item_latent
            prediction = latent_dot + user_b + item_b + global_mean
            if np.abs(actual_rating - prediction) < 0. or np.abs(actual_rating - prediction) > 5.0:  # Print only 10% of the time
                print(f"latent: {user_latent} {item_latent}")
                print(f"User ID: {user_id}, Product ID: {product_id}, loss {(actual_rating - prediction):.2f}, rating {actual_rating:.2f}, pred {prediction:.2f}, {latent_dot:.2f}, {user_b:.2f}, {item_b:.2f}, {global_mean:.2f}")

            # Calculate regularization terms
            user_regularization = np.sum(training_result.user_weights[user_id] ** 2)
            item_regularization = np.sum(training_result.item_weights[product_id] ** 2)
            user_metadata_regularization = np.sum(training_result.user_metadata_weights[user_id] ** 2)
            item_metadata_regularization = np.sum(training_result.item_metadata_weights[product_id] ** 2)
            user_bias_regularization = training_result.user_bias[user_id] ** 2
            item_bias_regularization = training_result.item_bias[product_id] ** 2
            
            regularization_term = self.regularization * (
                user_regularization + item_regularization + 
                user_metadata_regularization + item_metadata_regularization +
                user_bias_regularization + item_bias_regularization
            )
            
            total_loss += (actual_rating - prediction) ** 2 + regularization_term
        return (total_loss / len(training_data), actual_rating - prediction, regularization_term)

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
        user_idx: int,
        item_idx: int,
        global_mean: float, 
        user_latent_weights: NDArray[np.float64], 
        prod_latent_weights: NDArray[np.float64],
        user_bias: NDArray[np.float64],
        item_bias: NDArray[np.float64],
        user_metadata_weights: NDArray[np.float64],
        prod_metadata_weights: NDArray[np.float64],
        regularization: float,
        row: NDArray[np.float64],
        user_metadata_range: range = None,
        product_metadata_range: range = None
    ) -> Tuple[float, float]:
        """Compute the loss function."""
        # Calculate the latent factor contribution
        user_metadata_contribution = user_metadata_weights[user_idx] @ row[user_metadata_range]
        item_metadata_contribution = prod_metadata_weights[product_idx] @ row[product_metadata_range]

        user_latent_vector = user_latent_weights[user_idx] + user_metadata_contribution
        item_latent_vector = prod_latent_weights[product_idx] + item_metadata_contribution
        
        latent = np.dot(user_latent_vector, item_latent_vector)

        # Find user and item biases
        user_b = user_bias[user_idx]
        item_b = item_bias[product_idx]

        # Calculate regularization terms
        metadata_regularization = np.sum(user_metadata_weights ** 2) + np.sum(prod_metadata_weights ** 2)
        latent_regularization = np.sum(user_latent_weights[user_idx] ** 2) + np.sum(prod_latent_weights[product_idx] ** 2)
        bias_regularization = np.sum(user_bias ** 2) + np.sum(item_bias ** 2)
        regularization_term = regularization * (latent_regularization + bias_regularization + metadata_regularization)
        
        # Calculate residual and loss
        residual = actual_rating - (latent + user_b + item_b + global_mean)
        final_loss = residual ** 2 + regularization_term

        return final_loss

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

    def initialize_accumulator(self, n: int, n_metadata: int):
        grad = np.zeros(n_metadata)
        bias_acc = np.zeros(n)
        matrix = np.zeros(n, self.latent_factors, self.latent_factors)
        vector = np.zeors(n, self.latent_factor)
        return grad, bias_acc, matrix, vector
        
    def accumulate_update_latent(self,
        user_idx: int,
        product_idx: int,
        user_latent_weights: NDArray[np.float64],
        product_latent_weights: NDArray[np.float64],
        user_metadata_value: NDArray[np.float64],
        user_metadata_weights: NDArray[np.float64],
        item_metadata_value: NDArray[np.float64],
        item_metadata_weights: NDArray[np.float64],
        rating: float,
        global_mean: float,
        user_bias: NDArray[np.float64],
        product_bias: NDArray[np.float64],
    ):
        # Calculate user latent factor updates (for when fixing item factors)
        # A_u = outer(item_vector, item_vector), b_u = residual * item_vector
        item_vector = product_latent_weights[product_idx] + item_metadata_weights[product_idx] @ item_metadata_value
        residual_user = rating - global_mean - user_bias[user_idx] - product_bias[product_idx]
        new_user_latent_matrix = np.outer(item_vector, item_vector)
        new_user_latent_vector = residual_user * item_vector

        # Calculate product latent factor updates (for when fixing user factors)
        # A_i = outer(user_vector, user_vector), b_i = residual * user_vector
        user_vector = user_latent_weights[user_idx] + user_metadata_weights[user_idx] @ user_metadata_value
        residual_item = rating - global_mean - product_bias[product_idx] - user_bias[user_idx]
        new_product_latent_matrix = np.outer(user_vector, user_vector)
        new_product_latent_vector = residual_item * user_vector
        
        return new_user_latent_matrix, new_product_latent_matrix, new_user_latent_vector, new_product_latent_vector

    def accumulate_update_bias(self,
        user_idx: int,
        product_idx: int,
        rating: float,
        global_mean: float,
        user_bias: NDArray[np.float64],
        product_bias: NDArray[np.float64],
        user_latent_weights: NDArray[np.float64],
        product_latent_weights: NDArray[np.float64],
        user_metadata_value: NDArray[np.float64],
        user_metadata_weights: NDArray[np.float64],
        product_metadata_value: NDArray[np.float64],
        product_metadata_weights: NDArray[np.float64],
    ):
        # Calculate user bias update: residual - latent_contribution
        user_latent_contribution = user_latent_weights[user_idx] @ product_latent_weights[product_idx]
        user_metadata_contribution = (user_metadata_weights[user_idx] @ user_metadata_value) @ product_latent_weights[product_idx]
        product_metadata_contribution = (product_metadata_weights[product_idx] @ product_metadata_value) @ user_latent_weights[user_idx]
        
        new_user_bias = rating - global_mean - product_bias[product_idx] - user_latent_contribution - user_metadata_contribution

        # Calculate product bias update: residual - latent_contribution  
        new_product_bias = rating - global_mean - user_bias[user_idx] - user_latent_contribution - product_metadata_contribution

        return new_user_bias, new_product_bias

    def update_bias(self, 
        training_data: NDArray[np.float64],
        user_bias: NDArray[np.float64],
        item_bias: NDArray[np.float64],
        user_bias_acc: NDArray[np.float64],
        item_bias_acc: NDArray[np.float64],
        n_users: int,
        n_items: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        for user_id, user_idx in self.user_idx_map.items():
            n_items = len(training_data[training_data[:, 0] == user_id])
            user_bias[user_idx] = user_bias_acc[user_idx] / (n_items + self.regularization)
        for item_id, item_idx in self.product_idx_map.items():
            n_users = len(training_data[training_data[:, 1] == item_id])
            item_bias[item_idx] = item_bias_acc[item_idx] / (n_users + self.regularization)
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
        
        # Ensure gradients have the same shape as the weights
        if grad_user_metadata.shape != user_metadata_weights.shape:
            # Reshape or pad gradients to match weights shape
            if grad_user_metadata.size == 0:
                grad_user_metadata = np.zeros_like(user_metadata_weights)
            else:
                grad_user_metadata = np.resize(grad_user_metadata, user_metadata_weights.shape)
                
        if grad_prod_metadata.shape != prod_metadata_weights.shape:
            # Reshape or pad gradients to match weights shape
            if grad_prod_metadata.size == 0:
                grad_prod_metadata = np.zeros_like(prod_metadata_weights)
            else:
                grad_prod_metadata = np.resize(grad_prod_metadata, prod_metadata_weights.shape)

        # Gradient descent update: w = w - eta * (regularization * w - gradient)
        step_user = eta * (grad_user_metadata - self.regularization * user_metadata_weights)
        new_user_metadata_weights = user_metadata_weights + step_user

        step_prod = eta * (grad_prod_metadata - self.regularization * prod_metadata_weights)
        new_prod_metadata_weights = prod_metadata_weights + step_prod

        return new_user_metadata_weights, new_prod_metadata_weights

    def update_latent_weights(self,
        matrix_acc: NDArray[np.float64],
        vector_acc: NDArray[np.float64],
        current_weights: NDArray[np.float64],
        is_user: bool = True
    ) -> NDArray[np.float64]:
        """Update latent weights using accumulated matrices and vectors."""
        if is_user:
            index_map = self.user_idx_map
        else:
            index_map = self.product_idx_map
            
        for entity_id, entity_idx in index_map.items():
            A = matrix_acc[entity_idx] + self.regularization * np.eye(self.latent_factors)
            B = vector_acc[entity_idx]
            if np.linalg.cond(A) < 1e12:  # Check condition number
                current_weights[entity_idx] = np.linalg.solve(A, B)
            else:
                current_weights[entity_idx] = np.linalg.lstsq(A, B, rcond=None)[0]
        
        return current_weights

    def accumulate_metadata_gradients(self,
        user_idx: int,
        product_idx: int,
        residual: float,
        user_latent_weights: NDArray[np.float64],
        prod_latent_weights: NDArray[np.float64],
        user_metadata_range: range,
        product_metadata_range: range,
        row: NDArray[np.float64],
        grad_user_metadata: NDArray[np.float64],
        grad_prod_metadata: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Accumulate gradients for metadata weights."""
        # Gradient for user metadata weights: residual * item_vector * user_metadata  
        for k in range(len(user_metadata_range)):
            grad_user_metadata[user_idx, :, k] += residual * prod_latent_weights[product_idx] * row[user_metadata_range][k]
        
        # Gradient for product metadata weights: residual * user_vector * item_metadata
        for k in range(len(product_metadata_range)):
            grad_prod_metadata[product_idx, :, k] += residual * user_latent_weights[user_idx] * row[product_metadata_range][k]
            
        return grad_user_metadata, grad_prod_metadata


    def select_intersect(self, training_data: NDArray[np.float64], idd: int, column_select:int) -> NDArray[np.float64]:
        """
        Select a subset of training data based on intersection conditions.

        This method filters the training data to return only rows that satisfy certain
        intersection criteria while preserving all original columns.

        Args:
            training_data (NDArray[np.float64]): A 2D numpy array containing the training
                dataset with shape (n_samples, n_features). Each row represents a sample
                and each column represents a feature.

            id (int): The identifier used to filter the training data. This is typically
                the user or product ID that is used to select relevant rows.

            column_select (int): The index of the column in the training data that is used
                to determine the intersection condition. This column should contain the IDs
                that are relevant for filtering.

        Returns:
            NDArray[np.float64]: A filtered 2D numpy array containing only the rows that
                meet the intersection conditions. The array maintains the same number of
                columns as the input but may have fewer rows. Shape: (n_filtered_samples, n_features).

        Raises:
            ValueError: If the training_data is empty or has invalid dimensions.
            TypeError: If the training_data is not a numpy array of float64 type.
        Given a training data, select the subset of training data that only contains
        given conditions. Should return the same columns as the original training data.
        """

        if training_data.size == 0:
            raise ValueError("Training data is empty.")
        if not isinstance(training_data, np.ndarray) or training_data.dtype != np.float64:
            raise TypeError("Training data must be a numpy array of float64 type.")
        if column_select < 0 or column_select >= training_data.shape[1]:
            raise IndexError("Column index out of bounds.")

        # Select rows where the specified column matches the given id
        selected_rows = training_data[training_data[:, column_select] == idd]
        return selected_rows


class Trainer:
    """Main training class that manages the training process."""
    
    def __init__(self, hp: ALSHyperParameters):
        """Initialize the trainer with an ALS model."""
        print("param comb", hp.to_dict())
        self.hp = hp

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
            print("test results", dist)
            
        best_result = min(test_results, key=lambda x: x[1])
        return best_result[0]

    def get_data_from_indices(self, base_data, train_indices: List[RangeIndex], test_indices: RangeIndex):
        if not isinstance(train_indices, list):
            raise TypeError(f"Expected list of RangeIndex for train_indices, got {type(train_indices)}")
        if train_indices is None or len(train_indices) == 0:
            raise ValueError("train_indices must not be empty")
        tdata = np.concatenate([base_data[ti.start:ti.end] for ti in train_indices])
        vdata = base_data[test_indices.start:test_indices.end]
        return tdata, vdata
    