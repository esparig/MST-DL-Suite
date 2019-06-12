"""Test normalization inside Data Generator:
What method does take less?
Comparison of times between np.interp and /1023.

Results for 10 epochs:
    Using np.interp: 189.77499961853027 
    Using \1023: 173.77592372894287
"""
import random
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from keras.utils import Sequence
from keras.optimizers import SGD
from scipy.ndimage import shift

import sys
sys.path.append('..')
sys.path.append('.')

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.custom_model import get_model

class DataGenerator1(Sequence): # Testing purposes
    """Class Data Generator for Keras.
    """

    def __init__(self, image_filenames: List[str], labels: List[int],
                 batch_size: int, width_shift_range: float = 0):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.width_shift_range = width_shift_range

    def __len__(self) -> int:
        """Get the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        """Generates one batch of data
        """

        def _transform(img: np.ndarray,
                       transformation_x: float = 0, transformation_y: float = 0) -> np.ndarray:
            """Transform image as numpy tensor.
            Only width_shift and height_shift transformations implemented.
            """
            _, width, depth = img.shape
            shift_x = 0
            if transformation_x > 1.0 or transformation_x < -1.0:
                shift_x = 0
            else:
                rdm_shift_x = int(transformation_x * width)
                shift_x = random.randint(-rdm_shift_x, rdm_shift_x)

            if transformation_y > 1.0 or transformation_y < -1.0:
                shift_y = 0
            else:
                rdm_shift_y = int(transformation_y * width)
                shift_y = random.randint(-rdm_shift_y, rdm_shift_y)

            return np.stack(tuple(shift(img[:, :, i], (shift_y, shift_x),
                                        mode='nearest') for i in range(depth)), axis=-1)

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        result = (np.array([
            np.interp(
                _transform(np.load(file_name), transformation_x=self.width_shift_range),
                (0, 1023), (0, 1))
            for file_name in batch_x]), batch_y)

        return result

class DataGenerator2(Sequence): 
    """Class Data Generator for Keras.
    """

    def __init__(self, image_filenames: List[str], labels: List[int],
                 batch_size: int, width_shift_range: float = 0):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.width_shift_range = width_shift_range

    def __len__(self) -> int:
        """Get the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        """Generates one batch of data
        """

        def _transform(img: np.ndarray,
                       transformation_x: float = 0, transformation_y: float = 0) -> np.ndarray:
            """Transform image as numpy tensor.
            Only width_shift and height_shift transformations implemented.
            """
            _, width, depth = img.shape
            shift_x = 0
            if transformation_x > 1.0 or transformation_x < -1.0:
                shift_x = 0
            else:
                rdm_shift_x = int(transformation_x * width)
                shift_x = random.randint(-rdm_shift_x, rdm_shift_x)

            if transformation_y > 1.0 or transformation_y < -1.0:
                shift_y = 0
            else:
                rdm_shift_y = int(transformation_y * width)
                shift_y = random.randint(-rdm_shift_y, rdm_shift_y)

            return np.stack(tuple(shift(img[:, :, i], (shift_y, shift_x),
                                        mode='nearest') for i in range(depth)), axis=-1)

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        result = (np.array([
            _transform(np.load(file_name),
                       transformation_x=self.width_shift_range).astype('float32') / 1023
            for file_name in batch_x]), batch_y)

        return result

DS_PATH = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, _, _ = get_dataset(
    DS_PATH, percent_train=80, percent_val=20, percent_test=0)
print(len(X_TRAIN), len(X_VAL))
CLASSES = [folder.name for folder in DS_PATH.iterdir() if folder.is_dir()]
MODEL = get_model(input_shape=(200, 200, 24), CLASSES=len(CLASSES))
OPT = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
MODEL.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['acc', 'mse'])
GEN1_TRAIN = DataGenerator1(X_TRAIN, Y_TRAIN, 64)
GEN1_VAL = DataGenerator1(X_VAL, Y_VAL, 64)
GEN2_TRAIN = DataGenerator2(X_TRAIN, Y_TRAIN, 64)
GEN2_VAL = DataGenerator2(X_VAL, Y_VAL, 64)
t0 = time.time()
MODEL.fit_generator(generator=GEN1_TRAIN,
                    validation_data=GEN1_VAL,
                    epochs=10,
                    verbose=1)
t1 = time.time()
MODEL.fit_generator(generator=GEN2_TRAIN,
                    validation_data=GEN2_VAL,
                    epochs=10,
                    verbose=1)
t2 = time.time()
print("--------------------------\nUsing np.interp:", t1-t0, "\nUsing \\1023:", t2-t1)
