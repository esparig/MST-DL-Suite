"""Test data_generator.py
"""
import unittest
from pathlib import Path
from keras.optimizers import SGD

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.data_generator import DataGenerator
from multiscandllib.src.custom_model import get_model


class TestGetDataset(unittest.TestCase):
    """Test Class for data_generator.py
    """

    def setUp(self):
        self.ds_path = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
        x_train, y_train, x_val, y_val, _, _ = get_dataset(
            self.ds_path, percent_train=80, percent_val=20, percent_test=0)
        self.classes = [folder.name for folder in self.ds_path.iterdir() if folder.is_dir()]
        model = get_model(input_shape=(200, 200, 24), classes=len(self.classes))
        batch_size = 64
        num_epochs = 1
        opt = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])
        self.gen_train = DataGenerator(x_train, y_train, batch_size)
        self.gen_val = DataGenerator(x_val, y_val, batch_size)
        model.fit_generator(generator=self.gen_train,
                            validation_data=self.gen_val,
                            epochs=num_epochs,
                            verbose=1)

    def test_data_generator(self):
        """Test data generator
        """
        files_training = set(self.gen_train.store_batch)
        files_validation = set(self.gen_val.store_batch)

        self.assertEqual(len(files_training & files_validation), 0)

        set_from_folder = set()
        for category in self.classes:
            for file in self.ds_path.joinpath(category).iterdir():
                set_from_folder.add(file)

        self.assertEqual(len(files_training) + len(files_validation), len(set_from_folder))
        