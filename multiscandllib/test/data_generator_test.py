"""Test data_generator.py
"""
import unittest
from pathlib import Path

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.data_generator import DataGenerator


class TestGetDataset(unittest.TestCase):
    """Test Class for data_generator.py
    """

    def setUp(self):
        self.ds_path = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = get_dataset(
            self.ds_path,percent_train=80,percent_val=10,percent_test=10)
        self.classes = [folder.name for folder in self.ds_path.iterdir() if folder.is_dir()]
        train_gen = DataGenerator(self.x_train, self.y_train, 2, width_shift_range=0.2)
        val_gen = DataGenerator(self.x_val, self.y_val, 2)

    def test_data_generator(self):
        """Test data generator
        """
        #TO DO
