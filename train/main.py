#!/usr/bin/env python3
"""
Main executable for the train module.
Controls the training flow using scenario flags.
"""

import argparse
import sys
import os
from typing import Optional, List
import numpy as np
import json

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.io import DataIO
from train.validator import DataValidator
from train.pre import DataPreprocessorMovieLens
from train.train import Trainer
from intermediaries.dataclass import ALSHyperParameters, TrainingResult, dummy_training_result
import time
import traceback

np.set_printoptions(
    precision=2,        # Number of decimal places
    suppress=True,      # Suppress scientific notation
    linewidth=100,      # Characters per line
    formatter={'float': '{:6.2f}'.format}  # Custom formatting
)



class RecommendationSystemTrainer:
    """
    Main class that orchestrates the complete training pipeline.
    """

    def __init__(self, data_path: str = "dataset"):
        """Initialize the trainer with data path."""
        self.data_path = data_path
        self.data_io = DataIO(data_path)
    
    def train_recommendation_system(
        self, 
        test_ratio: float = 0.2,
        n_folds: int = 5,
        n_iter: List[int] = None,
        latent_factors: List[int] = None,
        regularization: List[float] = None
    ) -> TrainingResult:
        """
        Complete training pipeline for the recommendation system.
        
        Args:
            test_ratio: Ratio of data to use for testing
            n_folds: Number of folds for cross-validation
            n_iter: List of iteration counts to try
            latent_factors: List of latent factor counts to try
            regularization: List of regularization values to try
            
        Returns:
            TrainingResult: The best training result from hyperparameter search
        """

        prime_collector = {}
        
        # Set default hyperparameters if not provided
        if n_iter is None:
            n_iter = [4] # Default to 5 iterations for simplicity
        if latent_factors is None:
            latent_factors = [3]
        if regularization is None:
            regularization = [0.01]
        
        print("Step 1: Reading data...")
        # Read data using DataIO
        base_data = self.data_io.read_all()
        prime_collector["read_data"] = {
            "user_shape": base_data.user.shape,
            "product_shape": base_data.product.shape,
            "rating_shape": base_data.rating.shape
        }
        
        print("Step 2: Validating data...")
        # Validate data using DataValidator
        validator = DataValidator(base_data)
        validated_data = validator.validate_all()
        prime_collector["validated_data"] = {
            "user_shape": validated_data.user.shape,
            "product_shape": validated_data.product.shape,
            "rating_shape": validated_data.rating.shape
        }
        
        print("Step 3: Preprocessing data...")
        # Preprocess data using DataPreprocessorMovieLens
        preprocessor = DataPreprocessorMovieLens(validated_data)
        processed_data = preprocessor.process(validated_data, test_ratio, n_folds)
        prime_collector["processed_data"] = {
            "training_shape": processed_data.training_data.shape,
            "test_shape": processed_data.test_data.shape,
            "user_metadata_range": processed_data.user_metadata_range,
            "product_metadata_range": processed_data.product_metadata_range,
            "fold_count": len(processed_data.fold_indices)
        }
        print("Step 4: Training model...")
        # Create hyperparameters and train using Trainer
        hyperparameters = ALSHyperParameters(
            n_iter=n_iter,
            latent_factors=latent_factors,
            regularization=regularization
        )
        
        # Start timing
        start_time = time.perf_counter()
        
        trainer = Trainer(hyperparameters)
        best_result = trainer.find_best_parameters(processed_data)
        
        # End timing
        end_time = time.perf_counter()
        training_time = end_time - start_time
        
        prime_collector["best_result"] = {
            "parameters": best_result.parameters,
            "final_loss": best_result.final_loss,
            "time_taken": training_time
        }
        
        print("Step 5: Saving model...")
        self.data_io.save_training_result(best_result)
        return best_result

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train recommendation system')
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='dataset',
        help='Path to the dataset directory (default: dataset)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Ratio of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        nargs='+',
        default=[10],
        help='List of iteration counts to try (default: [5])'
    )
    
    parser.add_argument(
        '--latent-factors',
        type=int,
        nargs='+',
        default=[3],
        help='List of latent factor counts to try (default: [5])'
    )
    
    parser.add_argument(
        '--regularization',
        type=float,
        nargs='+',
        default=[.1],
        help='List of regularization values to try (default: [0.01])'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    try:
        args = parse_arguments()
        
        print("=== Recommendation System Training Pipeline ===")
        print(f"Data path: {args.data_path}")
        print(f"Test ratio: {args.test_ratio}")
        print(f"Number of folds: {args.n_folds}")
        print(f"Iterations to try: {args.n_iter}")
        print(f"Latent factors to try: {args.latent_factors}")
        print(f"Regularization values to try: {args.regularization}")
        print("=" * 50)
        
        # Create trainer and run the complete pipeline
        trainer = RecommendationSystemTrainer(args.data_path)
        result = trainer.train_recommendation_system(
            test_ratio=args.test_ratio,
            n_folds=args.n_folds,
            n_iter=args.n_iter,
            latent_factors=args.latent_factors,
            regularization=args.regularization
        )
        
        print("\n=== Training Complete ===")
        print(f"Best model saved with final loss: {result.final_loss}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
