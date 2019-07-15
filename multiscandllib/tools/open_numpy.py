"""Graphic representation of a numpy file
- The numpy file has to be a layered image with 4 channels: VISIBLE(H, S, I) + NIR
"""
import tkinter as tk
from tkinter import filedialog
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    """Opens a file explorer to be able to select a file
    """
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename()
    root.destroy()

    my_npy = np.load(file_path)
    _, _, layers = my_npy.shape # height, width, layers
    plt.rcParams.update({'font.size': 8})

    with sns.axes_style('dark'):
        # Get subplots
        rows = layers//4
        fig, axis = plt.subplots(rows, 4)
        axis[0, 0].set_title('H')
        axis[0, 1].set_title('S')
        axis[0, 2].set_title('I')
        axis[0, 3].set_title('NIR')
        for row in range(rows):
            # Display various LUTs
            col_h = axis[row, 0].imshow(my_npy[:, :, row*4], cmap=plt.cm.copper, aspect='equal') # pylint: disable=maybe-no-member
            fig.colorbar(col_h, ax=axis[row, 0])
            axis[row, 0].axis('off')

            col_s = axis[row, 1].imshow(my_npy[:, :, row*4+1], cmap=plt.cm.copper, aspect='equal') # pylint: disable=maybe-no-member
            fig.colorbar(col_s, ax=axis[row, 1])
            axis[row, 1].axis('off')

            col_i = axis[row, 2].imshow(my_npy[:, :, row*4+2], cmap=plt.cm.copper, aspect='equal') # pylint: disable=maybe-no-member
            fig.colorbar(col_i, ax=axis[row, 2])
            axis[row, 2].axis('off')

            col_nir = axis[row, 3].imshow(my_npy[:, :, row*4+3], cmap=plt.cm.gray, aspect='equal') # pylint: disable=maybe-no-member
            fig.colorbar(col_nir, ax=axis[row, 3])
            axis[row, 3].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
