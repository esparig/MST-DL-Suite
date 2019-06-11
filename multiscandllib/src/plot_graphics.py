"""
Plot performance graphics

"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_performance_graphics(history, num_epochs, preffix="",
                              output_folder=None, save_figure=True, show_figure=True):
    """Plot performance graphics of the trainig
    """
    
    plt.figure(0)
    plt.plot(history['acc'],'r')
    plt.plot(history['val_acc'],'g')
    if num_epochs > 19:
        plt.xticks(np.arange(0, num_epochs, num_epochs//10))
    else:
        plt.xticks(np.arange(0, num_epochs))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])

    if save_figure:
        plt.savefig(os.path.join(output_folder, preffix+"_acc.png"))
    if show_figure:
        plt.show()

    plt.figure(1)
    plt.plot(history['loss'],'r')
    plt.plot(history['val_loss'],'g')
    if num_epochs > 19:
        plt.xticks(np.arange(0, num_epochs, num_epochs//10))
    else:
        plt.xticks(np.arange(0, num_epochs))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])

    if save_figure:
        plt.savefig(os.path.join(output_folder, preffix+"_loss.png"))
    if show_figure:
        plt.show()

def plot_confusion_matrix(model, batch, labels, classes_names, preffix="",
                          output_folder=None, save_figure=True, show_figure=True):
    """Plot confusion matrix
    """
    model_predicted = np.argmax(model.predict_generator(batch, verbose=1), axis=1)
    model_df_cm = pd.DataFrame(
        confusion_matrix(np.argmax(labels, axis=1), model_predicted),
        classes_names, classes_names)

    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)  # for label size
    sns.heatmap(model_df_cm, annot=True, annot_kws={"size": 10})  # font size

    if save_figure:
        plt.savefig(os.path.join(output_folder,
                                 preffix+"_confusion_matrix.png"))
    if show_figure:
        plt.show()

    return model_predicted
