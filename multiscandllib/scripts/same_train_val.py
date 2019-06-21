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
        classes = args.classes
    else:
        classes = [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]

    # Get training, validation, and test datasets
    x_train, y_train, _, _, _, _ = get_dataset(dataset_path,
                                                       percent_train=100,
                                                       percent_val=0,
                                                       percent_test=0,
                                                       classes=classes)
    x_val = x_train[0:]
    y_val = y_train[0:]

    # Inizialize the model and print a summary
    model = get_model(input_shape=(200, 200, 24), classes=len(classes))
    model.summary()

    # Set hyperparameters, inizialize generators, and train the model
    batch_size = args.batch_size
    num_epochs = args.epochs

    opt = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    my_training_batch_generator = DataGenerator(x_train, y_train, batch_size)
    my_validation_batch_generator = DataGenerator(x_val, y_val, batch_size)

    # monitor = EarlyStopping(monitor='acc', patience=1)  # Not working as expected

    custom_model = model.fit_generator(generator=my_training_batch_generator,
                                       validation_data=my_validation_batch_generator,
                                       epochs=num_epochs,
                                       verbose=1)

    # Plot performance graphics
    current_datetime = str(datetime.datetime.now())

    plot_performance_graphics(custom_model.history, num_epochs, prefix=current_datetime,
                              output_folder=output_folder, show_figure=False)

    # Visualizing of confusion matrix
    custom_model_predicted = plot_confusion_matrix(model, my_validation_batch_generator, y_val,
                                                   classes, prefix=current_datetime,
                                                   output_folder=output_folder, show_figure=False)

    # Metrics: precision, recall, f1-score, support
    custom_model_report = classification_report(np.argmax(y_train, axis=1), custom_model_predicted)
    with open(os.path.join(output_folder, current_datetime+"_out.txt"), 'w') as file:
        file.write(custom_model_report)


if __name__ == "__main__":
    main()
