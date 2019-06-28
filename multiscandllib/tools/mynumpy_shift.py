import numpy as np
from scipy.ndimage import shift

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numpy(my_npy):
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

def transform(img, tx=0, ty=0): # only width_shift implemented
    heigh, width, depth = img.shape
    sx = 0
    if tx > 1.0 or tx < -1.0:
        sx = 0
    else:
        sx = int(tx*width)

    if ty > 1.0 or ty < -1.0:
        sy = 0
    else:
        sy = int(ty*width)

    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(img[:,:,0])
    # axarr[1].imshow(shift(img[:,:,0], (0, s), mode='nearest'))
    # plt.show()

    return np.stack(tuple(shift(img[:,:,i], (sy, sx), mode='nearest') for i in range(depth)), axis=-1)

def main():
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename()
    root.destroy()

    my_npy = np.load(file_path)
    # transform(my_npy, 0.5)
    plot_numpy(my_npy)
    plot_numpy(transform(my_npy, tx=0.5))


    return

if __name__ == "__main__":
    main()


# width = 4
# height = 3
# depth = 2
#
# img = np.zeros((height, width, depth), dtype=np.int16)
#
# for d in range(depth):
#     for i in range(height):
#         for j in range(width):
#             img[i][j][d] = (d+1)*10+j
#
# def print_image(img):
#     print(img.shape)
#     depth = img.shape[2]
#     for d in range(depth):
#         print(img[:,:,d])
# print(".........................................")
# print_image(img)
# print(".........................................")




# xs = transform(img, 0.5) #a number [-1.0, 1.0]
# print_image(xs)
# print(".........................................")
# xs = transform(img, -0.3) #a number [-1.0, 1.0]
# print_image(xs)
# print(".........................................")
# xs = transform(img, -1.3) #a number [-1.0, 1.0]
# print_image(xs)
# print(".........................................")
