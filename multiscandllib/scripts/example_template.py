"""Example using:

- custom_model_01: 7 blocks of Conv2D+BN+GN+MP

- dataset using 12 views (24 layers of depth)

- number of classes is 3

- batch ize is 64

- number of epochs is 10.

"""

import os
import datetime
import numpy as np

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

import sys

sys.path.append('..')
sys.path.append('.')

from src.get_dataset import get_dataset
from src.custom_model import get_model
from src.data_generator import DataGenerator
from src.plot_graphics import plot_performance_graphics, plot_confusion_matrix
from src.read_arguments import Argument, read_arguments

def main():
    """Main.
    
    Example of execution::

    python3 example.py --dataset <path to the dataset> --output <path to the output folder> --batch_size 64 --epochs 10 --lr <learning rate>
    """
    # Argument Parser
    args = read_arguments()

    # Get training, validation, and test datasets
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(args[Argument.DATASET],
                                                                 percent_train=80,
                                                                 percent_val=10,
                                                                 percent_test=10,
                                                                 classes=args[Argument.CLASSES])

    # Inizialize the model and print a summary
    model = get_model(input_shape=(200, 200, 24), classes=len(args[Argument.CLASSES]))
    model.summary()

    current_datetime = str(datetime.datetime.now())

    with open(os.path.join(args[Argument.OUTPUT], current_datetime+"_out.txt"), 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Set hyperparameters, inizialize generators, and train the model
    batch_size = args[Argument.BATCH_SIZE]
    num_epochs = args[Argument.EPOCHS]

    opt = SGD(lr=args[Argument.LR], decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    my_training_batch_generator = DataGenerator(x_train, y_train, batch_size)
    my_validation_batch_generator = DataGenerator(x_val, y_val, batch_size)

    checkpoint = ModelCheckpoint(
        os.path.join(args[Argument.OUTPUT], current_datetime+"_weights_{epoch:03d}_{val_acc:.4f}.hdf5"),
        monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]

    custom_model = model.fit_generator(generator=my_training_batch_generator,
                                       validation_data=my_validation_batch_generator,
                                       epochs=num_epochs,
                                       use_multiprocessing=False,
                                       workers=0, # If 0, will execute the generator on the main thread.
                                       callbacks=callback_list,
                                       verbose=1)

    # Plot performance graphics
    plot_performance_graphics(custom_model.history, num_epochs, prefix=current_datetime,
                              output_folder=args[Argument.OUTPUT], show_figure=False)

    # Visualizing of confusion matrix
    custom_model_predicted = plot_confusion_matrix(model, DataGenerator(x_test, y_test, batch_size),
                                                   y_test, args[Argument.CLASSES], prefix=current_datetime,
                                                   output_folder=args[Argument.OUTPUT], show_figure=False)

    # Metrics: precision, recall, f1-score, support
    custom_model_report = classification_report(np.argmax(y_test, axis=1), custom_model_predicted)
    with open(os.path.join(args[Argument.OUTPUT], current_datetime+"_out.txt"), 'w') as file:
        file.write(custom_model_report)

if __name__ == "__main__":
    main()
