"""Test Data Generator.
"""
import random
from typing import List, Tuple
from pathlib import Path
import numpy as np

from keras.utils import Sequence
from keras.optimizers import SGD
from scipy.ndimage import shift

import sys
sys.path.append('..')
sys.path.append('.')

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.custom_model import get_model

class DataGenerator2(Sequence): # Testing purposes
    """Class Data Generator for Keras.
    """

    def __init__(self, image_filenames: List[str], labels: List[int],
                 batch_size: int, width_shift_range: float = 0):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.width_shift_range = width_shift_range
        self.store_idx = [] # Testing purposes
        self.store_batch = [] # Testing purposes

    def __len__(self) -> int:
        """Get the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        """Generates one batch of data
        """
        self.store_idx.append(idx) # Testing purposes

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

        self.store_batch.extend(batch_x) # Testing purposes

        result = (np.array([
            _transform(np.load(file_name),
                       transformation_x=self.width_shift_range).astype('float32') / 1023
            for file_name in batch_x]), batch_y)

        return result

def test_data_generator():
    """Test data generator
    """
    files_training = set(GEN_TRAIN.store_batch)
    files_validation = set(GEN_VAL.store_batch)

    if not files_training & files_validation:
        print("TEST OK: examples from training and validation are different.")

    set_from_folder = set()
    for category in CLASSES:
        for file in DS_PATH.joinpath(category).iterdir():
            set_from_folder.add(file)

    # Not working when multiprocessing
    if len(files_training) + len(files_validation) == len(set_from_folder):
        print("TEST OK: Training and validation dataset contains all the files in the folder.")
    else:
        print("TEST FAIL: Datasets don't contain all the files in the folder.",
              len(files_training), len(files_validation), len(set_from_folder))

DS_PATH = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, _, _ = get_dataset(
    DS_PATH, percent_train=80, percent_val=20, percent_test=0)
print(len(X_TRAIN), len(X_VAL))
CLASSES = [folder.name for folder in DS_PATH.iterdir() if folder.is_dir()]
MODEL = get_model(input_shape=(200, 200, 24), CLASSES=len(CLASSES))
OPT = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
MODEL.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['acc', 'mse'])
GEN_TRAIN = DataGenerator2(X_TRAIN, Y_TRAIN, 64)
GEN_VAL = DataGenerator2(X_VAL, Y_VAL, 64)
MODEL.fit_generator(generator=GEN_TRAIN,
                    validation_data=GEN_VAL,
                    epochs=1,
                    verbose=1)

test_data_generator()
