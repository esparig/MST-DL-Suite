"""Script: training using fit function of keras.
"""
import sys
import os
import datetime
from keras.optimizers import SGD

sys.path.append('..')
sys.path.append('.')

from multiscandllib.src.get_dataset import serve_files, get_files
from src.read_arguments import Argument, read_arguments
from src.custom_model import get_model
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

    files, labels = get_files(arguments[Argument.DATASET], arguments[Argument.CLASSES], True)
    x_train, y_train = serve_files(files, labels, 1000)

    model = get_model(input_shape=(200, 200, 24), classes=len(arguments[Argument.CLASSES]))
    model.summary()

    current_datetime = str(datetime.datetime.now())

    with open(os.path.join(arguments[Argument.OUTPUT], current_datetime+"_out.txt"), 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    opt = SGD(lr=0.002, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])

    history = model.fit(x_train, y_train,
                        batch_size=arguments[Argument.BATCH_SIZE],
                        epochs=arguments[Argument.EPOCHS],
                        verbose=1,
                        validation_split=0.2)

    plot_performance_graphics(history.history,
                              arguments[Argument.EPOCHS],
                              prefix=current_datetime,
                              output_folder=arguments[Argument.OUTPUT],
                              show_figure=False)
    
    x_test, y_test = serve_files(files, labels, 100)
    plot_confusion_matrix(model,
                          x_test,
                          y_test,
                          arguments[Argument.CLASSES],
                          prefix=current_datetime,
                          output_folder=arguments[Argument.OUTPUT],
                          show_figure=False)
    