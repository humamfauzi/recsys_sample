"""
Data validation module for validating users, products, and ratings data.
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from intermediaries.dataclass import BaseData


class DataValidatorInterface(ABC):
    @abstractmethod
    def validate_all(self) -> BaseData:
        raise NotImplementedError("This method should be overridden by subclasses.")


class DataValidator(DataValidatorInterface):
    """
    Validates data integrity for recommendation system.
    Enclose it in a class to encapsulate validation methods while keeping the ability to
    have options for customization.
    """

    def __init__(self, basedata: BaseData):
        """Initialize the data validator."""
        self.users = basedata.user
        self.products = basedata.product
        self.ratings = basedata.rating

    def validate_users_exist_in_ratings(self) -> set[int]:
        """Validate that every user in rating table exists in user table."""
        user_ids_in_ratings: set[int] = set(self.ratings[:, 0].astype(int))
        all_user_ids: set[int] = set(self.users[:, 0].astype(int))
        missing_users = user_ids_in_ratings - all_user_ids
        return missing_users

    def validate_products_exist_in_ratings(self) -> set[int]:
        """Validate that every product in rating table exists in product table."""
        product_ids_in_ratings: set[int] = set(self.ratings[:, 1].astype(int))
        all_product_ids: set[int] = set(self.products[:, 0].astype(int))
        missing_products = product_ids_in_ratings - all_product_ids
        return missing_products

    def remove_null_ratings(self) -> np.ndarray:
        """Remove all null rating values."""
        cleaned_ratings = self.ratings[~np.isnan(self.ratings[:, 2].astype(float))]
        return cleaned_ratings

    def remove_null_release_year(self, ratings: np.ndarray) -> np.ndarray:
        """Remove products with null release year."""
        cleaned_products = self.products[self.products[:, 2] != ""]
        # Remove ratings for products that don't exist in cleaned products
        cleaned_product_ids = set(cleaned_products[:, 0].astype(int))
        cleaned_ratings = ratings[np.isin(ratings[:, 1].astype(int), list(cleaned_product_ids))]
        return cleaned_products, cleaned_ratings

    def validate_all(self) -> BaseData:
        """Run all validation checks and return cleaned ratings data."""
        if missing_users := self.validate_users_exist_in_ratings():
            raise ValueError(f"Missing users in ratings: {missing_users}")
        if missing_products := self.validate_products_exist_in_ratings():
            raise ValueError(f"Missing products in ratings: {missing_products}")
        cleaned_ratings = self.remove_null_ratings()
        products, cleaned_ratings = self.remove_null_release_year(cleaned_ratings)
        cleaned_ratings = cleaned_ratings.astype(float)
        return BaseData(rating=cleaned_ratings, user=self.users, product=products)
