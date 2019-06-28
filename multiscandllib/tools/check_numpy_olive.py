import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename()
    root.destroy()

    my_npy = np.load(file_path)
    heigh, width, layers = my_npy.shape
    plt.rcParams.update({'font.size': 8})

    with sns.axes_style('dark'):
        # Get subplots
        rows = layers//4
        fig, ax = plt.subplots(rows, 4)
        ax[0,0].set_title('H')
        ax[0,1].set_title('S')
        ax[0,2].set_title('I')
        ax[0,3].set_title('NIR')
        for l in range(rows):
            # Display various LUTs
            colH = ax[l,0].imshow(my_npy[:, :, l*4], cmap=plt.cm.copper, aspect='equal')
            fig.colorbar(colH, ax=ax[l,0])
            ax[l,0].axis('off')

            colS = ax[l,1].imshow(my_npy[:, :, l*4+1], cmap=plt.cm.copper, aspect='equal')
            fig.colorbar(colS, ax=ax[l,1])
            ax[l,1].axis('off')

            colI = ax[l,2].imshow(my_npy[:, :, l*4+2], cmap=plt.cm.copper, aspect='equal')
            fig.colorbar(colI, ax=ax[l,2])
            ax[l,2].axis('off')

            colNIR = ax[l,3].imshow(my_npy[:, :, l*4+3], cmap=plt.cm.gray, aspect='equal')
            fig.colorbar(colNIR, ax=ax[l,3])
            ax[l,3].axis('off')

    plt.show()
    return

if __name__ == "__main__":
    main()
