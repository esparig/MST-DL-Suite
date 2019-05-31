"""Example using:
- custom_model_01: 7 blocks of Conv2D+BN+GN+MP
- dataset using 12 views (24 layers of depth)
- number of classes is 3
- batch ize is 64
- number of epochs is 10.

Date: 20190531
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.custom_model import get_model
from multiscandllib.src.data_generator import DataGenerator
from multiscandllib.src.plot_graphics import plot_performance_graphics


def main():
    """Main function
    """
    # Argument Parser
    parser = argparse.ArgumentParser(description='Example DL training.')
    parser.add_argument('dataset', metavar='Dataset Path', type=Path, nargs='+',
                    help='a string with the dataset path')
    parser.add_argument('batch_size', metavar='Batch size', type=Path, nargs='+',
                    help='an integer for the batch size')
    parser.add_argument('epochs', metavar='Number of Epochs', type=Path, nargs='+',
                    help='an integer for the numer of epochs')
    args = parser.parse_args()

    # Get current script name
    output_folder = parser.prog
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set dataset path
    dataset_path = Path(args.dataset)

    # Get training, validation, and test datasets
    x_train, y_train, x_val, y_val, _, _ = get_dataset(dataset_path,
                                                       percent_train=80,
                                                       percent_val=20,
                                                       percent_test=0,
                                                       num_classes=3)
    """for i in range(5):
        print(x_train[i], y_train[i])
    """
    # Check an example
    # obj = np.load("G:\dataset\molestadoleve\ObjImg_20190306_164029.384_10407.npy")
    # print(obj)
    """
    # Inizialize and check the generator
    my_training_batch_generator = My_Generator(x_train, y_train, 2)
    x, y = my_training_batch_generator.__getitem__(0)
    print(x.shape)
    plot_numpy(x[0])
    plot_numpy(x[1])
    print(y.shape)


    # Inizialize and check the generator
    my_training_batch_generator = My_Generator(x_train, y_train, 2, width_shift_range = 0.2)
    x, y = my_training_batch_generator.__getitem__(0)
    print(x.shape)
    plot_numpy(x[0])
    plot_numpy(x[1])
    print(y.shape, "\n", y[:2])
    """

    # Inizialize the model and print a summary
    model = get_model(classes=3)
    model.summary()

    # Set hyperparameters, inizialize generators, and train the model
    batch_size = args.batch_size
    # num_training_samples = len(x_train)
    # num_validation_samples = len(x_test)

    opt = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    my_training_batch_generator = DataGenerator(x_train, y_train, batch_size, width_shift_range=0.2)
    my_validation_batch_generator = DataGenerator(x_val, y_val, batch_size)

    # monitor = EarlyStopping(monitor='acc', patience=1)  # Not working as expected

    num_epochs = args.epochs
    # steps_per_epoch = ceil(num_training_samples/batch_size)
    custom_model = model.fit_generator(generator=my_training_batch_generator,
            validation_data=my_validation_batch_generator,
            # steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            verbose=1)

    # Plot performance grpahics
    plot_performance_graphics(output_folder, custom_model, num_epochs)

    # Visualizing of confusion matrix
    custom_model_pred = model.predict_generator(my_validation_batch_generator, verbose=1)
    custom_model_predicted = np.argmax(custom_model_pred, axis=1)
    custom_model_cm = confusion_matrix(np.argmax(y_val, axis=1), custom_model_predicted)

    # classes_names = ['agostadograve', 'agostadoleve', 'granizo', 'molestadograve',
    # 'molestadoleve', 'molino', 'morada', 'picadodemosca', 'primera']
    classes_names = ['molestadograve', 'molestadoleve', 'primera']
    custom_model_df_cm = pd.DataFrame(custom_model_cm, classes_names, classes_names)
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)  # for label size
    sns.heatmap(custom_model_df_cm, annot=True, annot_kws={"size": 10})  # font size
    # plt.show()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))

    # Metrics: precision, recall, f1-score, support
    custom_model_report = classification_report(np.argmax(y_val, axis=1), custom_model_predicted)
    # print(simpleBlocks_report)
    with open(os.path.join(output_folder, 'out.txt'), 'w') as file:
        file.write(custom_model_report)


if __name__ == "__main__":
    main()
