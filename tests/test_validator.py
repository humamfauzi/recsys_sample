"""
Test cases for train/validator.py module.
"""

import unittest
import numpy as np
from train.validator import DataValidator
from intermediaries.dataclass import BaseData


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        users = np.array([[1, 'User1'], [2, 'User2']])
        products = np.array([[1, 'Product1'], [2, 'Product2']])

        ratings_failed_user = np.array([[1, 1, 5], [2, 2, 4], [3, 1, 3]])  # User ID 3 does not exist
        ratings_failed_product = np.array([[1, 1, 5], [2, 3, 4], [1, 2, 3]])  # Product ID 3 does not exist
        ratings_null = np.array([[1, 1, 5], [2, 2, np.nan], [1, 2, 3]])
        ratings_success = np.array([[1, 1, 5], [2, 2, 4], [1, 2, 3]])

        self.validator_success = DataValidator(BaseData(rating=ratings_success, user=users, product=products))
        self.validator_failed_user = DataValidator(BaseData(rating=ratings_failed_user, user=users, product=products))
        self.validator_failed_product = DataValidator(BaseData(rating=ratings_failed_product, user=users, product=products))
        self.validator_null_ratings = DataValidator(BaseData(rating=ratings_null, user=users, product=products))

    def test_validate_users_exist_in_ratings(self):
        """Test validation of users existence in ratings."""
        # Test implementation will go here
        missing = self.validator_success.validate_users_exist_in_ratings()
        self.assertEqual(missing, set())

        missing = self.validator_failed_user.validate_users_exist_in_ratings()
        self.assertEqual(missing, {3})  # User ID 3 does not exist
    
    def test_validate_products_exist_in_ratings(self):
        """Test validation of products existence in ratings."""
        # Test implementation will go here
        missing = self.validator_success.validate_products_exist_in_ratings()
        self.assertEqual(missing, set())

        missing = self.validator_failed_product.validate_products_exist_in_ratings()
        self.assertEqual(missing, {3})  # Product ID 3 does not exist

    def test_remove_null_ratings(self):
        """Test removal of null rating values."""
        # Test implementation will go here
        cleaned = self.validator_null_ratings.remove_null_ratings()
        expected = np.array([[1, 1, 5], [1, 2, 3]])
        np.testing.assert_array_equal(cleaned, expected)
    
    def test_validate_all(self):
        """Test complete validation pipeline."""
        # Test implementation will go here
        with self.assertRaises(ValueError):
            self.validator_failed_user.validate_all()
        with self.assertRaises(ValueError):
            self.validator_failed_product.validate_all()
        basedata = self.validator_null_ratings.validate_all()
        expected = np.array([[1, 1, 5], [1, 2, 3]])
        np.testing.assert_array_equal(basedata.rating, expected)

        with self.assertRaises(ValueError) as context:
            self.validator_failed_user.validate_all()
            self.assertTrue("Missing users in ratings: {3}" in str(context.exception))
        with self.assertRaises(ValueError) as context:
            self.validator_failed_product.validate_all()
            self.assertTrue("Missing products in ratings: {3}" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
