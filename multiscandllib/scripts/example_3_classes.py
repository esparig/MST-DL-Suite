"""Example using:

- custom_model_01: 7 blocks of Conv2D+BN+GN+MP

- dataset using 12 views (24 layers of depth)

- number of classes is 3

- batch ize is 64

- number of epochs is 10.

"""
import os
import argparse
import datetime
from pathlib import Path
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

def main():
    """Main function.
    """
    # Argument Parser
    parser = argparse.ArgumentParser(description='Example DL training.')
    parser.add_argument('--dataset', metavar='Dataset Path', type=str, required=True,
                        help='a string with the dataset path')
    parser.add_argument('--output', metavar='Output Path', type=str, required=True,
                        help='a string with the output path')
    parser.add_argument('--batch_size', metavar='Batch size', type=int, required=True,
                        help='an integer for the batch size')
    parser.add_argument('--epochs', metavar='Number of Epochs', type=int, required=True,
                        help='an integer for the numer of epochs')
    parser.add_argument('--classes', metavar='Selection of classes in the dataset', nargs='+',
                        help='classes without quotes, separated by a white space')
    args = parser.parse_args()

    print("Executing:", parser.prog)
    print("----------------------------------------")
    print("Dataset:", args.dataset)
    print("Output path:", args.output)
    print("Batch Size:", args.batch_size)
    print("Epochs:", args.epochs)


    # Get current script name
    output_folder = args.output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set dataset path
    dataset_path = Path(args.dataset)

    # Set classes
    if args.classes:
        classes = args.classes
    else:
        classes = [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]
    print("classes:", classes)

    # Get training, validation, and test datasets
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(dataset_path,
                                                                 percent_train=80,
                                                                 percent_val=10,
                                                                 percent_test=10,
                                                                 classes=classes)

    # Inizialize the model and print a summary
    model = get_model(input_shape=(200, 200, 24), classes=len(classes))
    model.summary()

    current_datetime = str(datetime.datetime.now())

    with open(os.path.join(output_folder, current_datetime+"_out.txt"), 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Set hyperparameters, inizialize generators, and train the model
    batch_size = args.batch_size
    num_epochs = args.epochs

    opt = SGD(lr=0.002, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    my_training_batch_generator = DataGenerator(x_train, y_train, batch_size)
    #, width_shift_range=0.2)
    my_validation_batch_generator = DataGenerator(x_val, y_val, batch_size)
    #my_validation_data = (load_dataset(x_val), y_val)

    # monitor = EarlyStopping(monitor='acc', patience=1)  # Not working as expected
    checkpoint = ModelCheckpoint(
        os.path.join(output_folder, current_datetime+"_weights_{epoch:03d}_{val_acc:.4f}.hdf5"),
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
                              output_folder=output_folder, show_figure=False)

    # Visualizing of confusion matrix
    custom_model_predicted = plot_confusion_matrix(model, DataGenerator(x_test, y_test, batch_size),
                                                   y_test, classes, prefix=current_datetime,
                                                   output_folder=output_folder, show_figure=False)

    # Metrics: precision, recall, f1-score, support
    custom_model_report = classification_report(np.argmax(y_test, axis=1), custom_model_predicted)
    with open(os.path.join(output_folder, current_datetime+"_out.txt"), 'w') as file:
        file.write(custom_model_report)

if __name__ == "__main__":
    main()
