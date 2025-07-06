"""
Test cases for train/io.py module.
"""

import unittest
import numpy as np
from train.io import DataIO
import os
import tempfile
import shutil
import json
from intermediaries.dataclass import TrainingResult


class TestDataIOInterface(unittest.TestCase):
    """Test cases for DataIOInterface."""
    
    def test_interface_methods_exist(self):
        """Test that interface methods are defined."""
        interface = DataIO()
        
        # Check that all required methods exist
        self.assertTrue(hasattr(interface, 'read_users'))
        self.assertTrue(hasattr(interface, 'read_products'))
        self.assertTrue(hasattr(interface, 'read_ratings'))
        self.assertTrue(hasattr(interface, 'save_model'))
        

class TestDataIO(unittest.TestCase):
    """Test cases for DataIO implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Create temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp(prefix='test_data_dummy_')
        
        # Create dummy CSV files for testing
        self.users_file = os.path.join(self.test_data_dir, 'user')
        users_data = ""
        users_data += "1|25|technician|21304\n"
        users_data += "2|50|executive|21304\n"
        users_data += "3|100|administrator|35234\n"
        users_data += "4|1|lawyer|22348\n"
        users_data += "5|4|programmer|33342\n"

        with open(self.users_file, 'w') as f:
            f.write(users_data)
        
        self.products_file = os.path.join(self.test_data_dir, 'item')
        products_data = ""
        products_data += "1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        products_data += "2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0\n"
        products_data += "3|Four Rooms (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0\n"
        products_data += "4|Get Shorty (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995)|0|1|0|0|0|1|0|0|1|0|0|0|0|0|0|0|0|0|0\n"
        products_data += "5|Copycat (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Copycat%20(1995)|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|0|1|0|0\n"
        products_data += "6|Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)|01-Jan-1995||http://us.imdb.com/Title?Yao+a+yao+yao+dao+waipo+qiao+(1995)|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0\n"
        with open(self.products_file, 'w') as f:
            f.write(products_data)
        
        self.ratings_file = os.path.join(self.test_data_dir, 'rating')
        # Ratings data somewhat has different separator, using tab instead of pipe
        # This is the actual data
        ratings_data = ""
        ratings_data += "196	242	3	881250949\n"
        ratings_data += "186	302	3	891717742\n"
        ratings_data += "22	377	1	878887116\n"
        ratings_data += "244	51	2	880606923\n"
        ratings_data += "166	346	1	886397596\n"
        ratings_data += "298	474	4	884182806\n"
        with open(self.ratings_file, 'w') as f:
            f.write(ratings_data)
        
        # Initialize DataIO instance
        self.data_io = DataIO(self.test_data_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary directory and all its contents
        if hasattr(self, 'test_data_dir') and os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_read_users_with_dummy_data(self):
        """Test reading users data with dummy data."""
        users = self.data_io.read_users()
        # Check that data is loaded correctly
        self.assertIsInstance(users, np.ndarray)
        self.assertEqual(users.shape[0], 5)  # 5 users in test data
        self.assertEqual(users.shape[1], 4)  # 4 columns: user_id, age, occupation, zip_code

        # Check that the data contains expected values
        first_user = users[0]
        self.assertEqual(first_user[0], '1')  # user_id
        self.assertEqual(first_user[1], '25')  # age
        self.assertEqual(first_user[2], 'technician')  # occupation
        self.assertEqual(first_user[3], '21304')  # zip_code

        last_user = users[-1]
        self.assertEqual(last_user[0], '5')  # user_id
        self.assertEqual(last_user[1], '4')  # age
        self.assertEqual(last_user[2], 'programmer')  # occupation
        self.assertEqual(last_user[3], '33342')  # zip_code
    
    def test_read_products_with_dummy_data(self):
        """Test reading products data with dummy data."""
        # Test implementation will go here
        products = self.data_io.read_products()
        # Check that data is loaded correctly
        self.assertIsInstance(products, np.ndarray)
        self.assertEqual(products.shape[0], 6)
        self.assertEqual(products.shape[1], 24)  # Assuming 24 columns based

        first_product = products[0]
        self.assertEqual(first_product[0], '1')  # product_id
        self.assertEqual(first_product[1], 'Toy Story (1995)')
        self.assertEqual(first_product[2], '01-Jan-1995')
        # skip third because it is an URL
        self.assertEqual(first_product[5], '0')
        self.assertEqual(first_product[6], '0')
        self.assertEqual(first_product[7], '0')
        self.assertEqual(first_product[8], '1')
        self.assertEqual(first_product[9], '1')
        self.assertEqual(first_product[10], '1')
        self.assertEqual(first_product[11], '0')
        self.assertEqual(first_product[12], '0')
        self.assertEqual(first_product[13], '0')
        self.assertEqual(first_product[14], '0')
        self.assertEqual(first_product[15], '0')
        self.assertEqual(first_product[16], '0')
        self.assertEqual(first_product[17], '0')
        self.assertEqual(first_product[18], '0')
        self.assertEqual(first_product[19], '0')
        self.assertEqual(first_product[20], '0')
        self.assertEqual(first_product[21], '0')
        self.assertEqual(first_product[22], '0')
        self.assertEqual(first_product[23], '0')

        last_product = products[-1]
        self.assertEqual(last_product[0], '6')  # product_id
        self.assertEqual(last_product[1], 'Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)')
        self.assertEqual(last_product[2], '01-Jan-1995')
        # skip third because it is an URL
        self.assertEqual(last_product[5], '0')
        self.assertEqual(last_product[6], '0')
        self.assertEqual(last_product[7], '0')
        self.assertEqual(last_product[8], '0')
        self.assertEqual(last_product[9], '0')
        self.assertEqual(last_product[10], '0')
        self.assertEqual(last_product[11], '0')
        self.assertEqual(last_product[12], '0')
        self.assertEqual(last_product[13], '1')
        self.assertEqual(last_product[14], '0')
        self.assertEqual(last_product[15], '0')
        self.assertEqual(last_product[16], '0')
        self.assertEqual(last_product[17], '0')
        self.assertEqual(last_product[18], '0')
        self.assertEqual(last_product[19], '0')
        self.assertEqual(last_product[20], '0')
        self.assertEqual(last_product[21], '0')
        self.assertEqual(last_product[22], '0')
        self.assertEqual(last_product[23], '0')
    
    def test_read_ratings_with_dummy_data(self):
        """Test reading ratings data with dummy data."""
        # Test implementation will go here
        ratings = self.data_io.read_ratings()
        self.assertIsInstance(ratings, np.ndarray)
        self.assertEqual(ratings.shape[0], 6)
        self.assertEqual(ratings.shape[1], 4)

        first_rating = ratings[0]
        self.assertEqual(first_rating[0], '196')  # user_id
        self.assertEqual(first_rating[1], '242')  # product_id
        self.assertEqual(first_rating[2], '3')    # rating
        self.assertEqual(first_rating[3], '881250949')  # timestamp

        last_rating = ratings[-1]
        self.assertEqual(last_rating[0], '298')  # user_id
        self.assertEqual(last_rating[1], '474')  # product_id
        self.assertEqual(last_rating[2], '4')    # rating
        self.assertEqual(last_rating[3], '884182806')  # timestamp

    def test_save_model(self):
        """Test saving model data."""
        # Test implementation will go here
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        self.data_io.save_model(arr)
        # Check if the file exists
        model_file_path = os.path.join(self.test_data_dir, 'model.npy')
        self.assertTrue(os.path.exists(model_file_path))

        # Check if the data is saved correctly
        self.assertTrue(np.array_equal(np.load(model_file_path), arr))
    
    def test_save_training_result(self):
        """Test saving training result with all components."""
        # Create mock training result data
        user_weights = np.random.rand(10, 5)
        item_weights = np.random.rand(20, 5)
        user_metadata_weights = np.random.rand(10, 3)
        item_metadata_weights = np.random.rand(20, 4)
        user_bias = np.random.rand(10)
        item_bias = np.random.rand(20)
        
        training_result = TrainingResult(
            parameters={'n_iter': 100, 'latent_factors': 5, 'regularization': 0.01},
            user_weights=user_weights,
            item_weights=item_weights,
            user_metadata_weights=user_metadata_weights,
            item_metadata_weights=item_metadata_weights,
            user_bias=user_bias,
            item_bias=item_bias,
            user_index_map={1: 0, 2: 1, 3: 2},
            product_index_map={101: 0, 102: 1, 103: 2},
            global_mean=3.5,
            final_loss=0.245
        )
        
        # Save the training result
        self.data_io.save_training_result(training_result)
        
        # Check that the directory structure is created
        # The directory name format is YYYYMMDD_XX where XX is a 2-char random string
        directories = [d for d in os.listdir(self.test_data_dir) if os.path.isdir(os.path.join(self.test_data_dir, d))]
        self.assertEqual(len(directories), 1)
        
        result_dir = os.path.join(self.test_data_dir, directories[0])
        
        # Check that all expected files exist
        expected_files = [
            'metadata.json',
            'user_weights.npy',
            'item_weights.npy',
            'user_bias.npy',
            'item_bias.npy',
            'user_metadata_weights.npy',
            'item_metadata_weights.npy'
        ]
        
        for filename in expected_files:
            file_path = os.path.join(result_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"File {filename} should exist")
        
        # Verify metadata.json content
        metadata_path = os.path.join(result_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('train_id', metadata)
        self.assertEqual(len(metadata['train_id']), 2)  # 2-character random string
        self.assertEqual(metadata['parameters'], training_result.parameters)
        # JSON converts integer keys to strings, so we need to convert them back for comparison
        expected_user_index_map = {str(k): v for k, v in training_result.user_index_map.items()}
        expected_product_index_map = {str(k): v for k, v in training_result.product_index_map.items()}
        self.assertEqual(metadata['user_index_map'], expected_user_index_map)
        self.assertEqual(metadata['product_index_map'], expected_product_index_map)
        self.assertEqual(metadata['global_mean'], training_result.global_mean)
        self.assertEqual(metadata['final_loss'], training_result.final_loss)
        
        # Verify numpy arrays are saved correctly
        loaded_user_weights = np.load(os.path.join(result_dir, 'user_weights.npy'))
        loaded_item_weights = np.load(os.path.join(result_dir, 'item_weights.npy'))
        loaded_user_bias = np.load(os.path.join(result_dir, 'user_bias.npy'))
        loaded_item_bias = np.load(os.path.join(result_dir, 'item_bias.npy'))
        loaded_user_metadata_weights = np.load(os.path.join(result_dir, 'user_metadata_weights.npy'))
        loaded_item_metadata_weights = np.load(os.path.join(result_dir, 'item_metadata_weights.npy'))
        
        np.testing.assert_array_equal(loaded_user_weights, user_weights)
        np.testing.assert_array_equal(loaded_item_weights, item_weights)
        np.testing.assert_array_equal(loaded_user_bias, user_bias)
        np.testing.assert_array_equal(loaded_item_bias, item_bias)
        np.testing.assert_array_equal(loaded_user_metadata_weights, user_metadata_weights)
        np.testing.assert_array_equal(loaded_item_metadata_weights, item_metadata_weights)
        
        # Clean up: remove the created directory
        shutil.rmtree(result_dir)
    
    def test_scalability_10_to_10000_rows(self):
        """Test that if it can read 10 rows, it can read 10000 rows."""
        # Test implementation will go here
        pass


if __name__ == '__main__':
    unittest.main()
