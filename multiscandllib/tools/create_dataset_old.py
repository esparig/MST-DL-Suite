import os
import sys
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image as iu

def create_dataset():
    X = [] # objects
    Y1 = [] # labels
    Y2 = [] # labels as integer categories
    dataset = "flowers"
    #dataset = "testds"
    cat = 0
    for currentdir in os.listdir(dataset):
        absdir = os.path.join(dataset,currentdir)
        if os.path.isdir(absdir):
            for f in os.listdir(absdir):
                #print(f)
                cimg = iu.load_img(os.path.join(absdir, f), target_size=(200, 200))
                #plt.imshow(cimg)
                #plt.show()
                A = iu.img_to_array(cimg)
                #B = np.einsum('ijk->kij', A)
                X.append(A)
                Y1.append(currentdir)
                Y2.append(cat)
            cat += 1

    size = len(Y2)

    perm = np.random.permutation(len(Y2))

    x_all = np.array(X)
    y_all = np.array(Y2)

    x_all = x_all[perm]
    y_all = y_all[perm]

    size_10 = size // 10

    p1 = size - 2 * size_10
    p2 = size - size_10

    x_train = x_all[0:p1]
    y_train = y_all[0:p1]
    x_val = x_all[p1:p2]
    y_val = y_all[p1:p2]
    x_test = x_all[p2:]
    y_test = y_all[p2:]

    print(x_train.shape, x_val.shape, x_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)


    np.save(os.path.join(dataset, 'x_train'), x_train)
    np.save(os.path.join(dataset, 'y_train'), y_train)
    np.save(os.path.join(dataset, 'x_val'), x_val)
    np.save(os.path.join(dataset, 'y_val'), y_val)
    np.save(os.path.join(dataset, 'x_test'), x_test)
    np.save(os.path.join(dataset, 'y_test'), y_test)

def get_10_layers(layers):
    size = len(layers)
    result = []
    if size == 10:
        return layers
    if size == 9:
        result.append(layers[0])
        result.extend(layers[:])
    if size == 8:
        result.append(layers[0])
        result.extend(layers[:5])
        result.extend(layers[4:])
    if size == 7:
        result.append(layers[0])
        result.extend(layers[:5])
        result.extend(layers[4:])
        result.append(layers[6])
    if size == 6:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[1])
        result.append(layers[2])
        result.append(layers[2])
        result.append(layers[3])
        result.append(layers[4])
        result.append(layers[4])
        result.append(layers[5])
        result.append(layers[5])
    if size == 5:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[2])
        result.append(layers[2])
        result.append(layers[3])
        result.append(layers[3])
        result.append(layers[4])
        result.append(layers[4])
    if size == 4:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[2])
        result.append(layers[2])
        result.append(layers[2])
        result.append(layers[3])
        result.append(layers[3])
    if size == 3:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[2])
        result.append(layers[2])
        result.append(layers[2])
    if size == 2:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[1])
        result.append(layers[1])
    if size == 1:
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
        result.append(layers[0])
    print(len(result), len(result[0]), len(result[0][0]), len(result[0][0][0]))
    return result

def main():
    if len(sys.argv) < 2:
        print("USAGE: python create_dataset.py <dir_path>")
        return

    dir_path = sys.argv[1]
    if os.path.isdir(dir_path):
        for i, fn in enumerate(glob.glob(os.path.join(dir_path, "*.json"))):
            obj = fn.split("_INFO")[0]
            imgHSI = []
            for v in range(10):
                if os.path.exists(obj+"_"+str(v)+"_H.png") and os.path.exists(obj+"_"+str(v)+"_S.png") and os.path.exists(obj+"_"+str(v)+"_I.png"):
                    imgH = cv2.imread(obj+"_"+str(v)+"_H.png", 2)
                    imgS = cv2.imread(obj+"_"+str(v)+"_S.png", 2)
                    imgI = cv2.imread(obj+"_"+str(v)+"_I.png", 2)
                    imgHSI.append(np.stack((imgH, imgS, imgI), axis=-1))
            print("Objeto:", i+1, "Views:", len(imgHSI), "\n", obj)
            # after this I need to rearrange views if total_views < 10 (duplicate some of them)
            print(imgHSI[0].shape)
            input_object = np.stack(get_10_layers(imgHSI), axis=-1)
            print(input_object.shape)
            print(input_object[:][:][:][0].shape)
            print(input_object[:][:][:][1].shape)
    else:
        raise Error("[E] dir_path not valid")

if __name__ == "__main__":
    main()
