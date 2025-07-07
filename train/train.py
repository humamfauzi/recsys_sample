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

    def fit(self, training_data: NDArray, user_metadata_range: range, product_metadata_range: range) -> TrainingResult:
        """Train the ALS model on training data."""
        self.user_idx_map, self.product_idx_map, n_users, n_items = self.find_unique_indices(training_data)
        # TODO: use enum to determine the column indices
        global_mean = np.mean(training_data[:, 2])  # Assuming ratings are in the third column
        user_latent_weights, prod_latent_weights, user_bias, item_bias = self.initialize_weights_and_bias(n_users, n_items)
        user_metadata_weights, prod_metadata_weights = self.initialize_metadata_weights(n_users, n_items, user_metadata_range, product_metadata_range)
        self.loss_iter_pair = []

        for iteration in range(self.n_iter):
            # prepare for accumlator
            # accumulator are required so I only iterate training data once in every iteration
            grad_user_metadata = np.zeros_like(user_metadata_weights)
            grad_prod_metadata = np.zeros_like(prod_metadata_weights)

            user_bias_acc, item_bias_acc = np.zeros(n_users), np.zeros(n_items)
            (user_latent_matrix_acc, 
            item_latent_matrix_acc, 
            user_latent_vector_acc, 
            item_latent_vector_acc) = self.initialize_latent_accumulator(n_users, n_items)
            
            current_loss = 0.0
            for row in training_data:
                user_id, product_id, actual_rating = row[0], row[1], row[2]
                loss, residual = self.loss_function(
                    global_mean,
                    user_latent_weights,
                    prod_latent_weights,
                    user_bias,
                    item_bias,
                    user_metadata_weights,
                    prod_metadata_weights,
                    regularization=self.regularization,
                    row=row,
                    user_metadata_range=user_metadata_range,
                    product_metadata_range=product_metadata_range
                )
                current_loss += loss
                
                # Accumulate user latent factor updates
                user_idx = self.user_idx_map[int(user_id)]
                product_idx = self.product_idx_map[int(product_id)]
                
                (user_latent_matrix_update, _, user_latent_vector_update, _) = self.accumulate_update_latent(
                    user_idx, product_idx, user_latent_weights, prod_latent_weights,
                    row[user_metadata_range], user_metadata_weights,
                    row[product_metadata_range], prod_metadata_weights,
                    actual_rating, global_mean, user_bias, item_bias
                )
                
                user_latent_matrix_acc[user_idx] += user_latent_matrix_update
                user_latent_vector_acc[user_idx] += user_latent_vector_update
                
                # Accumulate gradients for user metadata weights using the function
                grad_user_metadata, _ = self.accumulate_metadata_gradients(
                    user_idx, product_idx, residual, user_latent_weights, prod_latent_weights,
                    user_metadata_range, product_metadata_range, row, grad_user_metadata, grad_prod_metadata
                )
            
            # Update user latent weights
            user_latent_weights = self.update_latent_weights(
                user_latent_matrix_acc, user_latent_vector_acc, user_latent_weights, is_user=True
            )
            
            # ALS Step 2: Update item factors (fix user factors)
            (_, prod_latent_matrix_acc, _, prod_latent_vector_acc) = self.initialize_latent_accumulator(n_users, n_items)
            
            for row in training_data:
                user_id, product_id, actual_rating = row[0], row[1], row[2]
                
                # Accumulate item latent factor updates
                user_idx = self.user_idx_map[int(user_id)]
                product_idx = self.product_idx_map[int(product_id)]
                
                (_, item_latent_matrix_update, _, item_latent_vector_update) = self.accumulate_update_latent(
                    user_idx, product_idx, user_latent_weights, prod_latent_weights,
                    row[user_metadata_range], user_metadata_weights,
                    row[product_metadata_range], prod_metadata_weights,
                    actual_rating, global_mean, user_bias, item_bias
                )
                
                prod_latent_matrix_acc[product_idx] += item_latent_matrix_update
                prod_latent_vector_acc[product_idx] += item_latent_vector_update
                
                # Accumulate gradients for product metadata weights using the function
                _, grad_prod_metadata = self.accumulate_metadata_gradients(
                    user_idx, product_idx, residual, user_latent_weights, prod_latent_weights,
                    user_metadata_range, product_metadata_range, row, grad_user_metadata, grad_prod_metadata
                )
            
            # Update product latent weights
            prod_latent_weights = self.update_latent_weights(
                prod_latent_matrix_acc, prod_latent_vector_acc, prod_latent_weights, is_user=False
            )
            
            # Update biases
            user_bias_acc, item_bias_acc = np.zeros(n_users), np.zeros(n_items)
            for row in training_data:
                user_id, product_id, actual_rating = row[0], row[1], row[2]
                user_idx = self.user_idx_map[int(user_id)]
                product_idx = self.product_idx_map[int(product_id)]
                
                user_bias_update, item_bias_update = self.accumulate_update_bias(
                    user_idx, product_idx, actual_rating, global_mean,
                    user_bias, item_bias, user_latent_weights, prod_latent_weights,
                    row[user_metadata_range], user_metadata_weights,
                    row[product_metadata_range], prod_metadata_weights
                )
                user_bias_acc[user_idx] += user_bias_update
                item_bias_acc[product_idx] += item_bias_update
            
            user_bias, item_bias = self.update_bias(
                training_data, user_bias, item_bias, user_bias_acc, item_bias_acc, n_users, n_items
            )
            
            # Update metadata weights (normalize gradients by number of examples)
            n_examples = len(training_data)
            grad_user_metadata = grad_user_metadata / n_examples
            grad_prod_metadata = grad_prod_metadata / n_examples
            
            user_metadata_weights, prod_metadata_weights = self.update_metadata_weights(
                current_loss, user_metadata_weights, prod_metadata_weights,
                grad_user_metadata, grad_prod_metadata, self.eta
            )
            print(f"Iteration: {iteration} {}") 
            self.loss_iter_pair.append((iteration, current_loss))
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
        user_metadata_range: range = None,
        product_metadata_range: range = None
    ) -> Tuple[float, float]:
        """Compute the loss function."""
        user, item, actual_rating = int(row[0]), int(row[1]), row[2]
        user_idx = self.user_idx_map.get(user, -1)
        if user_idx == -1:
            raise ValueError(f"User {user} not found in user index map.")

        product_idx = self.product_idx_map.get(item, -1)
        if product_idx == -1:
            raise ValueError(f"Product {item} not found in product index map.")

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

    def initialize_latent_accumulator(self, 
        n_users: int, 
        n_items: int
    ): 
        user_latent_matrix_acc = np.zeros((n_users, self.latent_factors, self.latent_factors))
        prod_latent_matrix_acc = np.zeros((n_items, self.latent_factors, self.latent_factors))
        user_latent_vector_acc = np.zeros((n_users, self.latent_factors))
        prod_latent_vector_acc = np.zeros((n_items, self.latent_factors))
        return user_latent_matrix_acc, prod_latent_matrix_acc, user_latent_vector_acc, prod_latent_vector_acc
        
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
