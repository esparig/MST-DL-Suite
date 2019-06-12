"""Script to test fit_generator:

Using same train/validtion set.

Date: 20190611
"""
import os
import argparse
import datetime
from pathlib import Path
import numpy as np

from keras.optimizers import SGD
from sklearn.metrics import classification_report

import sys
sys.path.append('..')
sys.path.append('.')
from src.get_dataset import get_dataset
from src.custom_model import get_model
from src.data_generator import DataGenerator
from src.plot_graphics import plot_performance_graphics, plot_confusion_matrix


def main():
    """Main function
    """
    # Argument Parser
    parser = argparse.ArgumentParser(description='Example DL training.')
    parser.add_argument('--dataset', metavar='Dataset Path', type=str, required=True,
                        help='a string with the dataset path')
    parser.add_argument('--batch_size', metavar='Batch size', type=int, required=True,
                        help='an integer for the batch size')
    parser.add_argument('--epochs', metavar='Number of Epochs', type=int, required=True,
                        help='an integer for the numer of epochs')
    parser.add_argument('--CLASSES', metavar='Selection of CLASSES in the dataset', nargs='+',
                        help='CLASSES without quotes, separated by a white space')
    args = parser.parse_args()

    print("Executing:", parser.prog)
    print("----------------------------------------")
    print("Dataset:", args.dataset)
    print("Batch Size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Classes:", args.classes)

    # Get current script name
    output_folder = parser.prog[:-3]+".results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set dataset path
    dataset_path = Path(args.dataset)

    # Set CLASSES
    if args.classes:
        CLASSES = args.classes
    else:
        CLASSES = [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]

    # Get training, validation, and test datasets
    X_TRAIN, Y_TRAIN, _, _, _, _ = get_dataset(dataset_path,
                                                       percent_train=100,
                                                       percent_val=0,
                                                       percent_test=0,
                                                       CLASSES=CLASSES)
    X_VAL = X_TRAIN[0:]
    Y_VAL = Y_TRAIN[0:]

    # Inizialize the MODEL and print a summary
    MODEL = get_model(input_shape=(200, 200, 24), CLASSES=len(CLASSES))
    MODEL.summary()

    # Set hyperparameters, inizialize generators, and train the MODEL
    batch_size = args.batch_size
    num_epochs = args.epochs

    opt = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
    MODEL.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    my_training_batch_generator = DataGenerator(X_TRAIN, Y_TRAIN, batch_size)
    my_validation_batch_generator = DataGenerator(X_VAL, Y_VAL, batch_size)

    # monitor = EarlyStopping(monitor='acc', patience=1)  # Not working as expected

    custom_model = MODEL.fit_generator(generator=my_training_batch_generator,
                                       validation_data=my_validation_batch_generator,
                                       epochs=num_epochs,
                                       verbose=1)

    # Plot performance graphics
    current_datetime = str(datetime.datetime.now())

    plot_performance_graphics(custom_model.history, num_epochs, preffix=current_datetime,
                              output_folder=output_folder, show_figure=False)

    # Visualizing of confusion matrix
    custom_model_predicted = plot_confusion_matrix(MODEL, my_validation_batch_generator, Y_VAL,
                                                   CLASSES, preffix=current_datetime,
                                                   output_folder=output_folder, show_figure=False)

    # Metrics: precision, recall, f1-score, support
    custom_model_report = classification_report(np.argmax(Y_TRAIN, axis=1), custom_model_predicted)
    with open(os.path.join(output_folder, current_datetime+"_out.txt"), 'w') as file:
        file.write(custom_model_report)


if __name__ == "__main__":
    main()
