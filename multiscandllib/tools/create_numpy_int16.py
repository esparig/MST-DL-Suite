import collections
import os
import json
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

def check_views(data_src, category, id, json_fn, num_views=5):
    views_info = None
    with open(json_fn) as f:
        json_data = json.load(f)
        views_info = json_data["views"]
    list_layers = []

    for i in range(num_views):
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_H.png")
        if not os.path.exists(cur_layer):
            return (False, cur_layer+" NOT FOUND")
        list_layers.append(cur_layer)
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_S.png")
        if not os.path.exists(cur_layer):
            return (False, cur_layer+" NOT FOUND")
        list_layers.append(cur_layer)
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_I.png")
        if not os.path.exists(cur_layer):
            return (False, cur_layer+" NOT FOUND")
        list_layers.append(cur_layer)
        if "I" in views_info[0]:
            cur_id = id+"_"+str(i+1)+"2_I.png"
        else:
            cur_id = id+"_"+str(i)+"2_I.png"
        cur_layer = os.path.join(data_src, category, cur_id)
        if not os.path.exists(cur_layer):
            return (False, cur_layer+" NOT FOUND")
        list_layers.append(cur_layer)
    return (True, list_layers)

def list_views(data_src, category, id, json_fn, max_views=6):
    """
    views_info = None
    with open(json_fn) as f:
        json_data = json.load(f)
        views_info = json_data["views"]
    """
    list_layers = []
    for i in range(max_views):
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_H.png")
        if not os.path.exists(cur_layer):
            list_layers.append(None)
        else:
            list_layers.append(cur_layer)
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_S.png")
        if not os.path.exists(cur_layer):
            list_layers.append(None)
        else:
            list_layers.append(cur_layer)
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"1_I.png")
        if not os.path.exists(cur_layer):
            list_layers.append(None)
        else:
            list_layers.append(cur_layer)
        cur_layer = os.path.join(data_src, category, id+"_"+str(i)+"2_I.png")
        if not os.path.exists(cur_layer):
            list_layers.append(None)
        else:
            list_layers.append(cur_layer)
    return list_layers, max_views*4-list_layers.count(None)

def get_nearest_indexes(n):
    indexes = []
    for i in range(4, 21, 4):
        if n-i > -1:
            indexes.append(n-i)
        if n+i < 24:
            indexes.append(n+i)
    return indexes

def gen_dataset_12views(data_src, data_dst, src_num_views, write=0):
    df = pd.read_csv(os.path.join(data_src, "dataset.csv"), sep=";", header=None, names=['UID', 'category', 'src_folder', 'date', 'coop'])
    if write == 1:
        cats = df.category.unique()
        for c in cats:
            if not os.path.exists(os.path.join(data_dst, c)):
                os.makedirs(os.path.join(data_dst, c))
    objects_ok_cat, objects_error_cat = [], []
    for idx, row in tqdm(df.iterrows()):
        json_fn = os.path.join(data_src, row['category'], row['UID']+"_INFO.json") #json file name
        ob_layers, ob_num_ok = list_views(data_src, row['category'], row['UID'], json_fn, max_views=6)
        if ob_num_ok > src_num_views-1:
            nones = [i for i, x in enumerate(ob_layers) if x==None]
            for n in nones:
                for j in get_nearest_indexes(n):
                    if j not in nones:
                        ob_layers[n] = ob_layers[j]
                        break
            nones = [i for i, x in enumerate(ob_layers) if x==None]
            if len(nones)==0:
                if write == 1:
                    ob = np.zeros((HEIGHT, WIDTH, 24), dtype=np.int16) # rows, columns, (HSI color + I nir)*nviews/2
                    for i_layer, f in enumerate(ob_layers):
                        layer = io.imread(f)
                        for pixel_col in range(WIDTH):
                            for pixel_row in range(HEIGHT):
                                ob[pixel_row][pixel_col][i_layer] = layer[pixel_row][pixel_col]
                    path = os.path.join(data_dst, row['category'], row['UID'])
                    np.save(path, ob)
                objects_ok_cat.append(row['category'])
            else:
                objects_error_cat.append(row['category'])
        else:
            objects_error_cat.append(row['category'])
    return collections.Counter(objects_ok_cat), collections.Counter(objects_error_cat)

data_src = os.path.join("G:\\", "procesados")
data_dst = os.path.join("G:\\", "dataset_24_from_16")

WIDTH, HEIGHT = 200, 200
print(gen_dataset_12views(data_src, data_dst, 16, write=1))

"""
df = pd.read_csv(os.path.join(data_src, "dataset.csv"), sep=";", header=None, names=['UID', 'category', 'src_folder', 'date', 'coop'])
cats = df.category.unique()
for c in cats:
    if not os.path.exists(os.path.join(data_dst, c)):
        os.makedirs(os.path.join(data_dst, c))

objects_ok_cat, objects_error_cat = [], []
for idx, row in tqdm(df.iterrows()):
    json_fn = os.path.join(data_src, row['category'], row['UID']+"_INFO.json") #json file name
    ob_layers, ob_num_ok = list_views(data_src, row['category'], row['UID'], json_fn, max_views=6)
    if ob_num_ok > 15:
        nones = [i for i, x in enumerate(ob_layers) if x==None]
        for n in nones:
            for j in get_nearest_indexes(n):
                if j not in nones:
                    ob_layers[n] = ob_layers[j]
        nones = [i for i, x in enumerate(ob_layers) if x==None]
        if len(nones)==0:
            objects_ok_cat.append(row['category'])
        else:
            objects_error_cat.append(row['category'])
    else:
        objects_error_cat.append(row['category'])
counter_objects_ok = collections.Counter(objects_ok_cat)
counter_objects_error = collections.Counter(objects_error_cat)
print(counter_objects_ok)
print(counter_objects_error)
"""
"""
data_src = os.path.join("G:\\", "procesados")
data_dst = os.path.join("G:\\", "dataset")

WIDTH, HEIGHT = 200, 200

df = pd.read_csv(os.path.join(data_src, "dataset.csv"), sep=";", header=None, names=['UID', 'category', 'src_folder', 'date', 'coop'])

#TODO crear carpetas de las categorias de acuerdo al csv leído
category = ['agostadograve', 'agostadoleve', 'granizo', 'molestadograve', 'molestadoleve', 'molino', 'morada', 'picadodemosca', 'primera', 'primeracoop']
for c in category:
    if not os.path.exists(os.path.join(data_dst, c)):
        os.makedirs(os.path.join(data_dst, c))

objects_ok_cat, objects_error_cat = [], []
num_views = 6
for idx, row in tqdm(df.iterrows()):
    json_fn = os.path.join(data_src, row['category'], row['UID']+"_INFO.json") #json file name
    object_ok, object_info = check_views(data_src, row['category'], row['UID'], json_fn, num_views=num_views)
    if object_ok:
        objects_ok_cat.append(row['category'])
        ob = np.zeros((HEIGHT, WIDTH, num_views*4), dtype=np.int16) # rows, columns, (HSI color + I nir)*nviews/2
        for i, f in enumerate(object_info):
            #print(f)
            layer = io.imread(f)
            for pixel_col in range(WIDTH):
                for pixel_row in range(HEIGHT):
                    ob[pixel_row][pixel_col][i] = layer[pixel_row][pixel_col]

        path = os.path.join(data_dst, row['category'], row['UID'])
        np.save(path, ob)
    else:
        objects_error_cat.append(row['category'])
    counter_objects_ok = collections.Counter(objects_ok_cat)
    counter_objects_error = collections.Counter(objects_error_cat)
print(counter_objects_ok)
print(counter_objects_error)
"""

"""
ob1 = np.zeros((200, 200, 20), dtype=np.int16) # rows, columns, (HSI color + I nir)*nviews/2
np.save('test1_numpy_int16', ob1)

ob2 = np.zeros((200, 200, 20)) # rows, columns, (HSI color + I nir)*nviews/2
np.save('test2_numpy_int16', ob2)
"""
