"""
Plot performance graphics

"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_graphics(output_folder, current_model, num_epochs,
                              save_figure=True, show_figure=True):
    """Plot performance graphics of the trainig
    """
    # TO DO: choose between show/save plots

    plt.figure(0)
    plt.plot(current_model.history['acc'],'r')
    plt.plot(current_model.history['val_acc'],'g')
    plt.xticks(np.arange(0, num_epochs, num_epochs//10))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])

    plt.figure(1)
    plt.plot(current_model.history['loss'],'r')
    plt.plot(current_model.history['val_loss'],'g')
    plt.xticks(np.arange(0, num_epochs, num_epochs%10))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])

    if save_figure:
        plt.savefig(os.path.join(output_folder, "loss.png"))
    if show_figure:
        plt.show()
