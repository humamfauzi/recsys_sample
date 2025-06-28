"""
Test cases for train/pre.py module.
"""

import unittest
import numpy as np
from train.pre import DataPreprocessorMovieLens, DataProcessorInterface
from intermediaries.dataclass import BaseData, ProcessedTrainingData, RangeIndex, Folds


class TestDataPreprocessorMovieLens(unittest.TestCase):
    """Test cases for DataPreprocessorMovieLens class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample user data: [user_id, age, profession, gender]
        self.sample_user = np.array([
            [1, 18, "engineer", "M"],
            [2, 33, "teacher", "F"],
            [3, 45, "doctor", "M"],
            [4, 18, "student", "F"],
            [5, 60, "engineer", "M"]
        ])
        
        # Create sample product data: [product_id, name, release_date, imdb_url, unknown, Action, Adventure, Animation]
        self.sample_product = np.array([
            [1, "Movie1", "01-01-1995", "http://imdb.com/1", 0, 1, 0, 0],
            [2, "Movie2", "15-05-2000", "http://imdb.com/2", 0, 0, 1, 0],
            [3, "Movie3", "25-12-2009", "http://imdb.com/3", 0, 0, 0, 1],
            [4, "Movie4", "10-06-2019", "http://imdb.com/4", 0, 1, 1, 0]
        ])
        
        # Create sample ratings data: [user_id, product_id, rating, timestamp]
        self.sample_ratings = np.array([
            [1, 1, 5, 978824268],
            [1, 2, 3, 978824269],
            [2, 1, 4, 978824270],
            [2, 3, 2, 978824271],
            [3, 2, 5, 978824272],
            [4, 4, 1, 978824273],
            [5, 1, 4, 978824274]
        ])
        
        # Create BaseData object
        self.base_data = BaseData(
            rating=self.sample_ratings,
            user=self.sample_user,
            product=self.sample_product
        )
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessorMovieLens(self.base_data)
    
    def tearDown(self):
        """Clean up after each test method."""
        self.preprocessor = None
        self.base_data = None
    
    def test_encode_profession(self):
        """Test profession encoding method."""
        result = self.preprocessor.encode_profession()
        
        # Check shape - should be (num_users, num_unique_professions)
        unique_professions = np.unique(self.sample_user[:, 2])
        expected_shape = (len(self.sample_user), len(unique_professions))
        self.assertEqual(result.shape, expected_shape)
        
        # Check data type
        self.assertEqual(result.dtype, int)
        
        # Check one-hot encoding correctness
        # First user is "engineer" - should have 1 in engineer column
        engineer_idx = np.where(unique_professions == "engineer")[0][0]
        self.assertEqual(result[0, engineer_idx], 1)
        
        # Check that each row sums to 1 (one-hot encoding)
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(len(self.sample_user)))
        
        # Check first and last values
        self.assertIn(result[0, 0], [0, 1])  # First value should be 0 or 1
        self.assertIn(result[-1, -1], [0, 1])  # Last value should be 0 or 1
    
    def test_encode_gender(self):
        """Test gender encoding method."""
        result = self.preprocessor.encode_gender()
        
        # Check shape - should be (num_users,)
        expected_shape = (len(self.sample_user),)
        self.assertEqual(result.shape, expected_shape)
        
        # Check data type
        self.assertEqual(result.dtype, int)
        
        # Check encoding correctness (M=1, F=0)
        self.assertEqual(result[0], 1)  # First user is "M"
        self.assertEqual(result[1], 0)  # Second user is "F"
        self.assertEqual(result[2], 1)  # Third user is "M"
        
        # Check all values are 0 or 1
        self.assertTrue(np.all(np.isin(result, [0, 1])))
        
        # Check first and last values
        self.assertIn(result[0], [0, 1])
        self.assertIn(result[-1], [0, 1])
    
    def test_encode_age(self):
        """Test age encoding method."""
        result = self.preprocessor.encode_age()
        
        # Check shape
        expected_shape = (len(self.sample_user),)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that ages are binned correctly
        # Age 25 should be in bin 1 (18-25)
        self.assertEqual(result[0], 1)
        # Age 35 should be in bin 2 (25-35)  
        self.assertEqual(result[1], 2)
        # Age 45 should be in bin 3 (35-50)
        self.assertEqual(result[2], 3)
        
        # Check that all values are valid bin indices
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 6))  # 6 bins total
        
        # Check first and last values
        self.assertGreaterEqual(result[0], 0)
        self.assertGreaterEqual(result[-1], 0)
    
    def test_encode_user_data(self):
        """Test user data encoding method."""
        result = self.preprocessor.encode_user_data()
        
        # Check shape - should include user_id + age + gender + profession_encoding
        unique_professions = np.unique(self.sample_user[:, 2])
        expected_cols = 1 + 1 + 1 + len(unique_professions)  # id + age + gender + professions
        expected_shape = (len(self.sample_user), expected_cols)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that user IDs are preserved in first column
        np.testing.assert_array_equal(result[:, 0], self.sample_user[:, 0].astype(int))
        
        # Check first and last row values
        self.assertEqual(result[0, 0], 1)  # First user ID
        self.assertEqual(result[-1, 0], 5)  # Last user ID
    
    def test_drop_name(self):
        """Test drop_name method."""
        test_array = np.array([
            [1, "name1", "other1"],
            [2, "name2", "other2"],
            [3, "name3", "other3"]
        ])
        
        result = self.preprocessor.drop_name(test_array)
        
        # Check that column 1 (name) is removed
        expected_shape = (test_array.shape[0], test_array.shape[1] - 1)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that first and last columns are preserved
        np.testing.assert_array_equal(result[:, 0], test_array[:, 0])
        np.testing.assert_array_equal(result[:, 1], test_array[:, 2])
    
    def test_drop_imdb(self):
        """Test drop_imdb method."""
        test_array = np.array([
            [1, "col1", "imdb_url", "col3"],
            [2, "col1", "imdb_url", "col3"],
            [3, "col1", "imdb_url", "col3"]
        ])
        
        result = self.preprocessor.drop_imdb(test_array)
        
        # Check that column 2 (imdb) is removed
        expected_shape = (test_array.shape[0], test_array.shape[1] - 1)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that columns are correctly preserved
        np.testing.assert_array_equal(result[:, 0], test_array[:, 0])
        np.testing.assert_array_equal(result[:, 1], test_array[:, 1])
        np.testing.assert_array_equal(result[:, 2], test_array[:, 3])
    
    def test_encode_release_year(self):
        """Test release year encoding method."""
        result = self.preprocessor.encode_release_year()
        
        # Check shape
        expected_shape = (len(self.sample_product),)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that years are binned correctly
        # 1995 should be in bin 0
        self.assertEqual(result[0], 0)
        # 2000 should be in bin 1
        self.assertEqual(result[1], 1)
        # 2010 should be in bin 2
        self.assertEqual(result[2], 2)
        # 2020 should be in bin 4
        self.assertEqual(result[3], 4)
        
        # Check that all values are valid bin indices
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 6))  # 6 bins total
    
    def test_encode_product_data(self):
        """Test product data encoding method."""
        result = self.preprocessor.encode_product_data()
        
        # Check that result has correct structure
        # Should include: product_id + release_year + genre_columns
        expected_cols = 1 + 1 + (self.sample_product.shape[1] - 5)  # id + year + genres
        expected_shape = (len(self.sample_product), expected_cols)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that product IDs are preserved
        np.testing.assert_array_equal(result[:, 0], self.sample_product[:, 0].astype(int))
        
        # Check first and last values
        self.assertEqual(result[0, 0], 1)  # First product ID
        self.assertEqual(result[-1, 0], 4)  # Last product ID
    
    def test_merge_arrays(self):
        """Test merge_arrays method."""
        result = self.preprocessor.merge_arrays(self.base_data)
        
        # Check that result has correct number of rows (same as ratings)
        self.assertEqual(result.shape[0], len(self.sample_ratings))
        
        # Check that result has correct structure
        # Should include: rating_data + user_data + product_data
        self.assertGreater(result.shape[1], self.sample_ratings.shape[1])
        
        # Check first and last rows
        self.assertIsNotNone(result[0, 0])  # First rating
        self.assertIsNotNone(result[-1, 0])  # Last rating
    
    def test_split_train_test(self):
        """Test train/test split method."""
        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        test_ratio = 0.25
        
        train_data, test_data_result = self.preprocessor.split_train_test(test_data, test_ratio)
        
        # Check sizes
        expected_test_size = int(len(test_data) * test_ratio)
        expected_train_size = len(test_data) - expected_test_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data_result), expected_test_size)
        
        # Check that total data is preserved
        self.assertEqual(len(train_data) + len(test_data_result), len(test_data))
        
        # Check shapes
        self.assertEqual(train_data.shape[1], test_data.shape[1])
        self.assertEqual(test_data_result.shape[1], test_data.shape[1])
    
    def test_shuffle_create_folds(self):
        """Test fold creation method."""
        test_data = np.array([[i, i+1, i+2] for i in range(10)])
        n_folds = 3
        
        result = self.preprocessor.shuffle_create_folds(test_data, n_folds)
        
        # Check that correct number of folds is created
        self.assertEqual(len(result), n_folds)
        
        # Check that each fold has correct structure
        for fold in result:
            self.assertIsInstance(fold, Folds)
            self.assertIsInstance(fold.TestIndex, RangeIndex)
            self.assertIsInstance(fold.TrainIndex, list)
            
            # Check that train indices don't include test index
            self.assertNotIn(fold.TestIndex, fold.TrainIndex)
        
        # Check first and last fold
        self.assertIsInstance(result[0], Folds)
        self.assertIsInstance(result[-1], Folds)
    
    def test_process_integration(self):
        """Test the complete process method integration."""
        test_ratio = 0.2
        n_folds = 3
        
        result = self.preprocessor.process(self.base_data, test_ratio, n_folds)
        
        # Check return type
        self.assertIsInstance(result, ProcessedTrainingData)
        
        # Check that all components exist
        self.assertIsInstance(result.training_data, np.ndarray)
        self.assertIsInstance(result.test_data, np.ndarray)
        self.assertIsInstance(result.fold_indices, list)
        
        # Check that fold indices has correct length
        self.assertEqual(len(result.fold_indices), n_folds)
        
        # Check data shapes are reasonable
        self.assertGreater(result.training_data.shape[0], 0)
        self.assertGreater(result.test_data.shape[0], 0)
        self.assertGreater(result.training_data.shape[1], 0)
        self.assertGreater(result.test_data.shape[1], 0)
        
        # Check that train and test data have same number of columns
        self.assertEqual(result.training_data.shape[1], result.test_data.shape[1])
        
        # Check first and last values in training data
        self.assertIsNotNone(result.training_data[0, 0])
        self.assertIsNotNone(result.training_data[-1, -1])


class TestDataProcessorInterface(unittest.TestCase):
    """Test cases for DataProcessorInterface abstract class."""
    
    def test_interface_cannot_be_instantiated(self):
        """Test that the interface cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DataProcessorInterface()
    
    def test_interface_method_not_implemented(self):
        """Test that subclasses must implement the process method."""
        
        class IncompleteProcessor(DataProcessorInterface):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteProcessor()


if __name__ == '__main__':
    unittest.main()
