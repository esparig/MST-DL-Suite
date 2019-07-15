import os
import random
from collections import defaultdict
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

"""
Creates filenames and label lists for the generator (train/validation/test datasets)

percent_train: default 80
percent_val: default 10
percent_test: default 10

return (training_ds, training_labels, val_ds, val_labels, test_ds, test_labels)
"""
def create_filenames_and_labels(dataset_path, percent_train=80, percent_val=10, percent_test=10):

    if (percent_train+percent_val+percent_test) != 100:
        percent_train, percent_val, percent_test = 80, 10, 10
        print("Default values for training/validation/test examples have been set (80/10/10)")

    d = defaultdict(list)
    for category in [ folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder)) ]:
        d[category].extend([ os.path.join(dataset_path, category, it) for it in os.listdir(os.path.join(dataset_path, category)) ])

    val_ds, test_ds, training_ds = [], [], []
    val_labels, test_labels, training_labels = [], [], []

    for icat, cat in enumerate(d.keys()):
        len_d = len(d[cat])

        limit_val = len_d*percent_val//100
        limit_test = len_d*percent_test//100

        val_ds.extend(d[cat][0:limit_val])
        test_ds.extend(d[cat][limit_val:limit_val+limit_test])
        training_ds.extend(d[cat][limit_val+limit_test:])

        val_labels.extend([icat]*limit_val)
        test_labels.extend([icat]*limit_test)
        training_labels.extend([icat]*(len_d-limit_val-limit_test))

    training_perm = np.random.permutation(len(training_labels))
    val_perm      = np.random.permutation(len(val_labels))
    test_perm     = np.random.permutation(len(test_labels))

    training_ds_np, training_labels_np = np.array(training_ds)[training_perm], np.array(training_labels)[training_perm]
    val_ds_np, val_labels_np =  np.array(val_ds)[val_perm],  np.array(val_labels)[val_perm]
    test_ds_np, test_labels_np =  np.array(test_ds)[test_perm],  np.array(test_labels)[test_perm]

    return (training_ds_np.tolist(), to_categorical(training_labels_np.tolist(), num_classes=9),
            val_ds_np.tolist(), to_categorical(val_labels_np.tolist(), num_classes=9),
            test_ds_np.tolist(), to_categorical(test_labels_np.tolist(), num_classes=9))

"""
Keras data generator
Using a Sequence is thread safe
"""
class My_Generator(Sequence): #width_shift_

    def __init__(self, image_filenames, labels, batch_size, seq):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.seq = seq

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x0 = np.array([np.load(file_name) for file_name in batch_x])
        x1 = np.array([np.load(file_name)/1023 for file_name in batch_x])
        x2 = self.test(x0[0])
        plt.imshow(x2)
        return (x0, x2, batch_y)

    def test(self, img):
        transform={}
        print(img.shape[1])
        if (self.seq.width_shift_range!=0.0):
            dx = self.seq.width_shift_range*img.shape[1]//100
            dx=random.uniform(-dx,dx)
            transform.update({'tx':dx})
        #if (self.seq.width_shift_range!=0.0):
            return self.seq.apply_transform(img[:,:,0:3], transform), dx

    def transform():

        transform={}
        #HORIZAONTAL FLIP
        flip=False
        if (args.da_flip_h):
            if random.randint(0,1):
                flip=True
                transform.update({'flip_horizontal':True})

        #HORIZAONTAL AND VERTICAL SHIFT
        dx=0
        dy=0
        if (args.da_width!=0.0)or(args.da_height!=0.0):
            dx=(args.da_width*args.width)//100
            dy=(args.da_height*args.height)//100
            dx=random.uniform(-dx,dx)
            dy=random.uniform(-dy,dy)
            transform.update({'tx':dx,'ty':dy})

        # SCALE
        scale=1.0
        if (args.da_zoom!=0.0):
            scale=random.uniform(1.0-args.da_zoom,1.0)
            transform.update({'zx':scale,'zy':scale})


        if (args.da_flip_h)or(args.da_width!=0.0)or(args.da_height!=0.0)or(args.da_zoom!=0.0):
            x=gen.apply_transform(x, transform)

        return x,dx,dy,scale,flip

def show(my_npy):
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

if __name__ == '__main__':
    # Set dataset path
    ds_path = os.path.join("G:\\", "dataset_24_from_16")
    x_train, y_train, _, _, _, _ = create_filenames_and_labels(ds_path, percent_train=80, percent_val=20, percent_test=0)
    image_data_gen = ImageDataGenerator(width_shift_range=0.5)
    sequence = My_Generator(x_train, y_train, 2, image_data_gen)
    x0, x1, _ = sequence.__getitem__(0)
    show(x0[0])
    show(x1[0])
    x0, x1, _ = sequence.__getitem__(1)
    show(x0[1])
    show(x1[1])
