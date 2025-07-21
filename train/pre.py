"""
Data preprocessing module for preparing data before training.
"""

import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod
from intermediaries.dataclass import BaseData, ProcessedTrainingData
from intermediaries.dataclass import RangeIndex, Folds

class DataProcessorInterface(ABC):
    @abstractmethod
    def process(self, basedata: BaseData, test_ratio: float, n_folds: int) -> ProcessedTrainingData:
        raise NotImplementedError("This method should be overridden by subclasses.")

class DataPreprocessorMovieLens(DataProcessorInterface):
    """
    Preprocesses data for training. Different dataset might have different preprocessing steps or event class.
    All class should implement this interface to ensure consistency.
    """
    
    def __init__(self, basedata: BaseData):
        """Initialize the data preprocessor."""
        self.basedata = basedata
        self.processed_training_data = ProcessedTrainingData(
            training_data=np.array([]),
            test_data=np.array([]),
            fold_indices=[],
            user_metadata_range=range(0, 0),
            product_metadata_range=range(0, 0)
        )

    def encode_profession(self, user_data: np.ndarray) -> np.ndarray:
        # TODO: Implement enumerate for column access
        uniq_profession = np.unique(user_data[:, 2])
        profession_encoded = np.zeros((len(user_data), len(uniq_profession)), dtype=int)
        for idx, profession in enumerate(uniq_profession):
            profession_encoded[user_data[:, 2] == profession, idx] = 1
        return profession_encoded

    def encode_gender(self, user_data: np.ndarray) -> np.ndarray:
        gender_encoded = np.zeros(len(user_data), dtype=int)
        # TODO: Implement enumerate for column access
        gender_encoded[user_data[:, 3] == "M"] = 1
        return gender_encoded

    def encode_age(self, user_data: np.ndarray) -> np.ndarray:
        """Encode age into bins."""
        # TODO: Implement enumerate for column access
        age = user_data[:, 1].astype(int)
        age_bins = np.digitize(age, bins=[0, 18, 25, 35, 50, 70])
        return age_bins - 1

    def encode_user_data(self, user_data: np.ndarray) -> np.ndarray:
        """Combine all user data into a single array."""
        stacked = np.hstack((
            user_data[:, 0].astype(int).reshape(-1, 1),  # User ID
            self.encode_age(user_data).reshape(-1, 1),
            self.encode_gender(user_data).reshape(-1, 1),
            self.encode_profession(user_data)
        ))
        return stacked

    def drop_name(self, product_arr: np.ndarray) -> np.ndarray:
        """Drop the name column from user data."""
        # TODO: Implement enumerate for column access
        return np.delete(product_arr, 1, axis=1)

    def drop_imdb(self, product_arr: np.ndarray) -> np.ndarray:
        """Drop the IMDb column from product data."""
        # TODO: Implement enumerate for column access
        return np.delete(product_arr, 3, axis=1)

    def drop_unknown(self, product_arr: np.ndarray) -> np.ndarray:
        """Drop the unknown column from product data."""
        # TODO: Implement enumerate for column access
        return np.delete(product_arr, 4, axis=1)

    def encode_release_year(self, product_arr: np.ndarray) -> np.ndarray:
        """Extract year from the timestamp in ratings."""
        # TODO: Implement enumerate for column access
        release_date = product_arr[:, 2]
        years = np.vectorize(lambda x: x.split("-")[-1])(release_date).astype(int)
        year_bins = np.digitize(years, bins=[1995, 2000, 2005, 2010, 2015, 2020])
        return year_bins - 1

    def encode_product_data(self, product_arr: np.ndarray) -> np.ndarray:
        """Combine all product data into a single array."""
        # TODO: Implement enumerate for column access
        release_year = self.encode_release_year(product_arr)
        metadata = product_arr[:, 5:]  # Assuming genre columns start from index 5
        product_data = self.drop_name(product_arr)
        product_data = self.drop_imdb(product_data)
        product_data = self.drop_unknown(product_data)
        product_data = np.hstack((
            product_data[:, 0].reshape(-1, 1),  # Product ID
            release_year.reshape(-1, 1),
            metadata,
        )).astype(int)
        return product_data

    def merge_arrays(self, basedata: BaseData) -> np.ndarray:
        """Merge all three different arrays into one large array."""
        encoded_user_data = self.encode_user_data(basedata.user)
        encoded_product_data = self.encode_product_data(basedata.product)
        rating_data = np.delete(basedata.rating, [-1], axis=1).astype(int)

        merged_columns = rating_data.shape[1] + encoded_user_data.shape[1] + encoded_product_data.shape[1]
        # add plus one in both metadata range to avoid picking id. We still need to keep the id for filtering purposes
        user_metadata_range = range(rating_data.shape[1] + 1, rating_data.shape[1] + encoded_user_data.shape[1])
        product_metadata_range = range(rating_data.shape[1] + encoded_user_data.shape[1] + 1, merged_columns)
        merged = np.zeros((rating_data.shape[0], merged_columns), dtype=int)

        for i in range(rating_data.shape[0]):
            user_id = rating_data[i, 0]
            product_id = rating_data[i, 1]
            
            rating_data_reshaped = rating_data[i, :].reshape(-1)
            filtered_encoded_user_data = encoded_user_data[encoded_user_data[:, 0] == user_id].reshape(-1)
            filtered_encoded_item_data = encoded_product_data[encoded_product_data[:, 0] == product_id].reshape(-1)

            merged[i, :] = np.hstack((
                rating_data_reshaped,
                filtered_encoded_user_data,
                filtered_encoded_item_data
            ))
        return merged, user_metadata_range, product_metadata_range

    def split_train_test(self, data: np.ndarray, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Separate data into training and test sets with randomization."""
        np.random.seed(42)
        data = data[np.random.permutation(data.shape[0])]
        split_index = int(len(data) * (1 - test_ratio))
        training_data = data[:split_index]
        test_data = data[split_index:]
        return training_data.astype(float), test_data.astype(float)
    
    def shuffle_create_folds(self, training_data: np.ndarray, n_folds: int) -> List[np.ndarray]:
        """Separate training data into folds with randomization."""
        np.random.seed(42)
        np.random.shuffle(training_data)
        fold_size = len(training_data) // n_folds
        range_index_collection = []
        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else len(training_data)
            ri = RangeIndex(start=start, end=end)
            range_index_collection.append(ri)
        folds_collection = []
        for ri in range_index_collection:
            train_indices = [other_ri for other_ri in range_index_collection if other_ri != ri]
            folds = Folds(test_index=ri, train_index=train_indices)
            folds_collection.append(folds)
        return folds_collection

    def process(self, basedata: BaseData, test_ratio: float, n_folds: int) -> ProcessedTrainingData:
        """Complete preprocessing pipeline."""
        merged, user_md_range, prod_md_range = self.merge_arrays(basedata)
        training_data, test_data = self.split_train_test(merged, test_ratio)
        folds = self.shuffle_create_folds(training_data, n_folds)
        return ProcessedTrainingData(
            training_data=training_data, 
            test_data=test_data, 
            fold_indices=folds,
            user_metadata_range=user_md_range,
            product_metadata_range=prod_md_range
        )
