"""Data Generation for Keras.
"""
import random
from typing import List, Tuple
import numpy as np
from keras.utils import Sequence
from scipy.ndimage import shift


class DataGenerator(Sequence):
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
