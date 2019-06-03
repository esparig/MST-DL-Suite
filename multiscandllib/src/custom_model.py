"""Custom Models:
Input shape: (200,200,24) to match our example shape (width, heigh, layers).
Layers: 24 = (3HSI(color)+1I(nir))*6 views.
Classes: 9 being [agostadograve, agostadoleve, granizo, molestadograve, molestadoleve,
molino, morada, picadodemosca, primera].
"""
from typing import Tuple

from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def _custom_model_01(input_shape: Tuple[int, int, int], classes: int) -> Sequential:
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


def get_model(input_shape: Tuple[int, int, int]=(200, 200, 24), classes: int=9) -> Sequential:
    """Caller function to create the Model.
    """
    return _custom_model_01(input_shape = input_shape, classes = classes)
