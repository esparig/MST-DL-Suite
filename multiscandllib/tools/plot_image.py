import os
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plt_object(path, category, id, view):
    try:
        imH = io.imread(os.path.join(path, category, id+"_"+str(view)+"_H.png"))
    except:
        imH = np.zeros((200, 200), dtype=np.uint16)
    try:
        imS = io.imread(os.path.join(path, category, id+"_"+str(view)+"_S.png"))
    except:
        imS = np.zeros((200, 200), dtype=np.uint16)
    imI = io.imread(os.path.join(path, category, id+"_"+str(view)+"_I.png"))
    imNIR = io.imread(os.path.join(path, category, id+"_"+str(view[0]+"2")+"_I.png"))

    with sns.axes_style('dark'):
        # Get subplots
        fig, ax = plt.subplots(2, 2, figsize=(8,8))

        # Display various LUTs
        colH = ax[0,0].imshow(imH, cmap=plt.cm.gray)
        fig.colorbar(colH, ax=ax[0,0])
        ax[0,0].set_title('H')
        colS = ax[0,1].imshow(imS, cmap=plt.cm.RdBu_r)
        fig.colorbar(colS, ax=ax[0,1])
        ax[0,1].set_title('S')
        colI = ax[1,0].imshow(imI, cmap=plt.cm.viridis)
        fig.colorbar(colI, ax=ax[1,0])
        ax[1,0].set_title('I')
        colM = ax[1,1].imshow(imNIR, cmap=plt.cm.copper)
        fig.colorbar(colM, ax=ax[1,1])
        ax[1,1].set_title('NIR')

    plt.show()

ds_path = os.path.join("G:\\", "procesados")
category = "primera"
#id = "ObjImg_20181105_133935.577_1209"
#category = "morada"

#id = "ObjImg_20190306_162837.253_11081"
#id = "ObjImg_20190305_152030.581_6207"
id = "ObjImg_20190306_214632.642_10244"
plt_object(ds_path, category, id, "11")
#for i in range(12):
    #plt_object(ds_path, category, id, i)
