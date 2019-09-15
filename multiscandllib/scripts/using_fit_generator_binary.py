"""Script: training using fit function of keras.
"""
import sys
import os
import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('..')
sys.path.append('.')

from src.get_dataset import serve_files, get_files
from src.read_arguments import Argument, read_arguments
from src.custom_model import custom_model_02
from src.plot_graphics import plot_performance_graphics, plot_confusion_matrix

def main():
    """Main function:
    - read_arguments
    - get_dataset
    - get_model
    - train
    - evaluate
    """
    arguments = read_arguments()
    if len(arguments[Argument.CLASSES]) > 2:
        print("Error: Number of classes should be 2.")
        raise ValueError

    files, labels = get_files(arguments[Argument.DATASET], arguments[Argument.CLASSES], False)
    train_len = int(len(files)*0.8)
    val_len = int(len(files)*0.1)

    x_train, y_train = files[:train_len], labels[:train_len]
    x_val, y_val = files[train_len:train_len+val_len], labels[train_len:train_len+val_len]
    x_test, y_test = files[train_len+val_len:], labels[train_len+val_len:]
    # x_train, y_train = next(serve_files(files, labels, arguments[Argument.EXAMPLES]))
    # x_train.shape should be (nb_sample, height, width, channel)
    # x_val, y_val = next(serve_files(files, labels, 200))

    print(x_train[0], np.load(x_train[0]).shape)
    model = custom_model_02(input_shape=(np.load(x_train[0]).shape))
    model.summary()

    current_datetime = str(datetime.datetime.now())

    with open(os.path.join(arguments[Argument.OUTPUT], current_datetime+"_out.txt"), 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # opt = SGD(lr=arguments[Argument.LR], decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_dataframe(
        pd.DataFrame(list(zip(x_train, y_train))),
        class_mode='binary')

    val_datagen = ImageDataGenerator()

    val_generator = val_datagen.flow_from_dataframe(
        pd.DataFrame(list(zip(x_val, y_val))),
        class_mode='binary')

    model.fit_generator(
        train_generator, np.argmax(y_train, axis=-1),
        epochs=arguments[Argument.EPOCHS],
        validation_data=val_generator)

    '''
    history = model.fit(x_train, np.argmax(y_train, axis=-1),
                        batch_size=arguments[Argument.BATCH_SIZE],
                        epochs=arguments[Argument.EPOCHS],
                        verbose=1,
                        validation_split=0.2)
    
    plot_performance_graphics(history.history,
                              arguments[Argument.EPOCHS],
                              prefix=current_datetime,
                              output_folder=arguments[Argument.OUTPUT],
                              show_figure=False)

    x_test, y_test = next(serve_files(files, labels, 200))
    plot_confusion_matrix(model,
                          x_test,
                          y_test,
                          arguments[Argument.CLASSES],
                          prefix=current_datetime,
                          output_folder=arguments[Argument.OUTPUT],
                          show_figure=False)
    '''

if __name__ == "__main__":
    main()
