"""Example using:
- custom_model_01: 7 blocks of Conv2D+BN+GN+MP
- dataset using 12 views (24 layers of depth)

Date: 20190613
"""

import random
from typing import List, Tuple

import numpy as np

from keras.layers import GaussianNoise
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.utils import Sequence

def get_model(input_shape: Tuple[int, int, int], classes: int) -> Sequential:
    """Custom model form by 7 blocks. Each block contains:
    - Conv2D
    - BatchNormalization
    - GaussianNoise
    - MaxPooling2D
    Followed by the classification block.
    """

    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", name='block1_conv1',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  # to 100x100

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))  # to 50x50

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))  # to 25x25

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))  # to 12x12

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))  # to 6x6

    # Block 6
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool'))  # to 3x3

    # Block 7
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv1'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool'))  # to 1x1

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    return model

class DummyGeneratorMemory(Sequence):
    """Class Data Generator for Keras.
    """
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Get the number of batches per epoch.
        """
        return 587

    def __getitem__(self, _: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generates one dummy batch of data, similar to the original dataset.
        """
        rdm_batch = np.array([np.random.rand(200, 200, 24).astype('float32')
                              for _ in range(self.batch_size)])
        rdm_labels = to_categorical([random.randint(0, 2) for _ in range(self.batch_size)], 3)

        return (rdm_batch, rdm_labels)

class DummyGenerator(Sequence):
    """Class Data Generator for Keras.
    """
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Get the number of batches per epoch.
        """
        return 587

    def __getitem__(self, _: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generates one dummy batch of data, similar to the original dataset.
        """
        rdm_batch = np.array([np.load("dummy.npy").astype('float32')
                              for _ in range(self.batch_size)])
        rdm_labels = to_categorical([random.randint(0, 2) for _ in range(self.batch_size)], 3)

        return (rdm_batch, rdm_labels)

def main():
    """Main function
    """

    # Inizialize the model and print a summary
    model = get_model(input_shape=(200, 200, 24), classes=3)
    model.summary()

    # Set hyperparameters
    batch_size = 64
    num_epochs = 100
    opt = SGD(lr=0.002, decay=1e-9, momentum=0.9, nesterov=True)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    # Create dummy file data and Inizialize generators
    dummy_data = np.random.rand(200, 200, 24).astype('int16')
    np.save("dummy.npy", dummy_data)

    dummy_generator = DummyGenerator(batch_size)

    # Train the model
    model.fit_generator(generator=dummy_generator,
                        validation_data=dummy_generator,
                        epochs=num_epochs,
                        use_multiprocessing=True,
                        workers=16,
                        verbose=1)

if __name__ == "__main__":
    main()
