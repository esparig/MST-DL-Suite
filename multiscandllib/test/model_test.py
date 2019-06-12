"""Test keras model
"""
import unittest
from pathlib import Path
from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.custom_model import get_model
from multiscandllib.src.data_generator import DataGenerator


class TestModel(unittest.TestCase):
    """Test Keras Model
    """

    def setUp(self):
        dsPath = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
        CLASSES = [folder.name for folder in dsPath.iterdir() if folder.is_dir()]
        X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, _, _ = get_dataset(
            dsPath, percent_train=80, percent_val=20, percent_test=0)
        self.model = get_model(input_shape=(200, 200, 24), CLASSES=len(CLASSES))
        self.model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['acc', 'mse'])
        my_training_batch_generator = DataGenerator(X_TRAIN, Y_TRAIN, 32, width_shift_range=0.2)
        my_validation_batch_generator = DataGenerator(X_VAL, Y_VAL, 32)
        self.custom_model = self.model.fit_generator(generator=my_training_batch_generator,
                                                validation_data=my_validation_batch_generator,
                                                epochs=1,
                                                verbose=1)
    def test_model(self):
        """Test shape of output wights, should have a 24 somewhere
        """
        block1_conv2d = self.model.layers[0]
        weights = block1_conv2d.get_weights()
        print(block1_conv2d.output_shape())
        self.assertEqual(len(weights[0][0][0]), 24)
