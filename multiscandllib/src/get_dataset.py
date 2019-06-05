"""
Generation of train, validation, and test datasets
"""
import os
from typing import List, Tuple, Sequence
from pathlib import Path
from collections import defaultdict
import numpy as np
from keras.utils import to_categorical

Dataset = Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]

def get_dataset(dataset_path: Path, percent_train: int=80, percent_val: int=10,
                percent_test: int=10, num_classes: int=9) -> Sequence[Dataset]:
    """
    Returns filename and label lists of samples

    percent_train: default 80
    percent_val: default 10
    percent_test: default 10

    return (training_ds, training_labels, val_ds, val_labels, test_ds, test_labels)
    """
    dataset_path = str(dataset_path) # TO DO: change everything to work with pathlib
    if (percent_train + percent_val + percent_test) != 100:
        percent_train, percent_val, percent_test = 80, 10, 10
        print("Default values for training/validation/test examples have been set (80/10/10)")

    files_by_category = defaultdict(list)
    for category in [ folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder)) ]:
        files_by_category[category].extend([ os.path.join(dataset_path, category, it) for it in
                                            os.listdir(os.path.join(dataset_path, category))])

    val_ds, test_ds, training_ds = [], [], []
    val_labels, test_labels, training_labels = [], [], []

    for icat, cat in enumerate(files_by_category.keys()):
        len_d = len(files_by_category[cat])

        limit_val = len_d * percent_val // 100
        limit_test = len_d * percent_test // 100

        val_ds.extend(files_by_category[cat][0:limit_val])
        test_ds.extend(files_by_category[cat][limit_val:limit_val + limit_test])
        training_ds.extend(files_by_category[cat][limit_val + limit_test:])

        val_labels.extend([icat] * limit_val)
        test_labels.extend([icat] * limit_test)
        training_labels.extend([icat] * (len_d - limit_val - limit_test))

    training_perm = np.random.permutation(len(training_labels))
    val_perm = np.random.permutation(len(val_labels))
    test_perm = np.random.permutation(len(test_labels))

    training_ds_np = np.array(training_ds)[training_perm]
    training_labels_np = np.array(training_labels)[training_perm]

    val_ds_np = np.array(val_ds)[val_perm]
    val_labels_np = np.array(val_labels)[val_perm]

    test_ds_np = np.array(test_ds)[test_perm]
    test_labels_np = np.array(test_labels)[test_perm]

    return (training_ds_np.tolist(),
            to_categorical(training_labels_np.tolist(), num_classes=num_classes),
            val_ds_np.tolist(), to_categorical(val_labels_np.tolist(), num_classes=num_classes),
            test_ds_np.tolist(), to_categorical(test_labels_np.tolist(), num_classes=num_classes))
