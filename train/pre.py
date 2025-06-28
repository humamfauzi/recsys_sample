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
            fold_indices=[]
        )

    def encode_profession(self) -> np.ndarray:
        # TODO: Implement enumerate for column access
        uniq_profession = np.unique(self.basedata.user[:, 2])
        profession_encoded = np.zeros((len(self.basedata.user), len(uniq_profession)), dtype=int)
        for idx, profession in enumerate(uniq_profession):
            profession_encoded[self.basedata.user[:, 2] == profession, idx] = 1
        return profession_encoded

    def encode_gender(self) -> np.ndarray:
        gender_encoded = np.zeros(len(self.basedata.user), dtype=int)
        # TODO: Implement enumerate for column access
        gender_encoded[self.basedata.user[:, 3] == "M"] = 1
        return gender_encoded

    def encode_age(self) -> np.ndarray:
        """Encode age into bins."""
        # TODO: Implement enumerate for column access
        age = self.basedata.user[:, 1].astype(int)
        age_bins = np.digitize(age, bins=[0, 18, 25, 35, 50, 70])
        return age_bins - 1

    def encode_user_data(self) -> np.ndarray:
        """Combine all user data into a single array."""
        user_data = np.hstack((
            self.basedata.user[:, 0].astype(int).reshape(-1, 1),  # User ID
            self.encode_age().reshape(-1, 1),
            self.encode_gender().reshape(-1, 1),
            self.encode_profession()
        ))
        return user_data

    def drop_name(self, product_arr: np.ndarray) -> np.ndarray:
        """Drop the name column from user data."""
        # TODO: Implement enumerate for column access
        return np.delete(product_arr, 1, axis=1)

    def drop_imdb(self, product_arr: np.ndarray) -> np.ndarray:
        """Drop the IMDb column from product data."""
        # TODO: Implement enumerate for column access
        return np.delete(product_arr, 2, axis=1)

    def encode_release_year(self) -> np.ndarray:
        """Extract year from the timestamp in ratings."""
        # TODO: Implement enumerate for column access
        release_date = self.basedata.product[:, 2]
        years = np.vectorize(lambda x: x.split("-")[-1])(release_date).astype(int)
        year_bins = np.digitize(years, bins=[1995, 2000, 2005, 2010, 2015, 2020])
        return year_bins - 1

    def encode_product_data(self) -> np.ndarray:
        """Combine all product data into a single array."""
        # TODO: Implement enumerate for column access
        product_data = np.copy(self.basedata.product)
        product_data = self.drop_name(product_data)
        product_data = self.drop_imdb(product_data)
        product_data = np.hstack((
            product_data[:, 0].reshape(-1, 1),  # Product ID
            self.encode_release_year().reshape(-1, 1),
            self.basedata.product[:, 5:]  # Genre columns (Action, Adventure, etc. as binary)
        )).astype(int)
        return product_data

    def merge_arrays(self, basedata: BaseData) -> np.ndarray:
        """Merge all three different arrays into one large array."""
        encoded_user_data = self.encode_user_data()
        encoded_product_data = self.encode_product_data()
        rating_data = np.delete(basedata.rating, [-1], axis=1)
        merged_columns = rating_data.shape[1] + encoded_user_data.shape[1] + encoded_product_data.shape[1]
        merged = np.zeros((rating_data.shape[0], merged_columns), dtype=int)
        for i in range(rating_data.shape[0]):
            user_id = rating_data[i, 0]
            product_id = rating_data[i, 1]
            merged[i, :] = np.hstack((
                rating_data[i, :].reshape(-1), 
                encoded_user_data[encoded_user_data[:, 0] == user_id].reshape(-1),
                encoded_product_data[encoded_product_data[:, 0] == product_id].reshape(-1)
            ))
        return merged

    def split_train_test(self, data: np.ndarray, test_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Separate data into training and test sets with randomization."""
        np.random.seed(42)
        data = data[np.random.permutation(data.shape[0])]
        split_index = int(len(data) * (1 - test_ratio))
        training_data = data[:split_index]
        test_data = data[split_index:]
        return training_data, test_data
    
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
            folds = Folds(TestIndex=ri, TrainIndex=train_indices)
            folds_collection.append(folds)
        return folds_collection

    def process(self, basedata: BaseData, test_ratio: float, n_folds: int) -> ProcessedTrainingData:
        """Complete preprocessing pipeline."""
        merged = self.merge_arrays(basedata)
        training_data, test_data = self.split_train_test(merged, test_ratio)
        folds = self.shuffle_create_folds(training_data, n_folds)
        return ProcessedTrainingData(training_data=training_data, test_data=test_data, fold_indices=folds)
