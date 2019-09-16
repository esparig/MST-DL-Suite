"""
.. module:: custom_model
    :synopsis: model creating functions
 
.. moduleauthor:: E. Parcero
"""
from typing import Tuple

from keras.models import Sequential
from keras.layers import GaussianNoise, Activation, Dropout
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def custom_model_01(input_shape: Tuple[int, int, int], classes: int) -> Sequential:
    """Create a custom model formed by 7 blocks.

    Each block contains:
    
    - Conv2D
    - BatchNormalization
    - GaussianNoise
    - MaxPooling2D

    After these blocks a classification block is added.

    Args:
        input_shape: The input shape (ex: (100, 100, 3))
        
        classes: Number of classes

    Returns:
        model: The created model
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

def custom_model_02(input_shape: Tuple[int, int, int]) -> Sequential:
    """Create a custom model for binary classification formed by 3 blocks.

    Each block contains:
    
    - Conv2D
    - MaxPooling2D

    After these blocks a classification block is added.

    Args:
        input_shape: The input shape (ex: (100, 100, 3))

    Returns:
        model: The created model
    """

    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), padding="same", name='block1_conv1', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  # to 100x100

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', name='block2_conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))  # to 50x50

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', name='block3_conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))  # to 25x25

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def get_model(input_shape: Tuple[int, int, int], classes: int) -> Sequential:
    return custom_model_01(input_shape, classes)