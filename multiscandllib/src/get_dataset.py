"""
Generation of train, validation, and test datasets
"""
import warnings
import random
from typing import List, Tuple, Sequence
from pathlib import Path
from collections import defaultdict
import numpy as np
from keras.utils import to_categorical

Dataset = Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]
Dataset_Tuple = Tuple[List[Path], List[List[int]]]


def get_dataset(dataset_path: Path, percent_train: int=80, percent_val: int=10,
                percent_test: int=10, classes: List[str]=None) -> Sequence[Dataset]:
    """
    Gets filename and label lists of samples

    percent_train: default 80
    percent_val: default 10
    percent_test: default 10

    return (training_ds, training_labels, val_ds, val_labels, test_ds, test_labels)
    """
    if (percent_train + percent_val + percent_test) != 100:
        percent_train, percent_val, percent_test = 80, 10, 10
        warnings.warn(
            "Default values for training/validation/test examples have been set (80/10/10)",
            Warning)

    files_by_category = defaultdict(list)
    if not classes:
        classes = [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]

    num_classes = len(classes)

    for category in classes:
        files_by_category[category].extend(
            [file for file in dataset_path.joinpath(category).iterdir()])

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

def get_files(dataset_path: Path, classes: List[str], balanced: bool) -> Dataset_Tuple:
    """
    Get a list of files for our dataset.
    Arguments:
    - dataset_path: A Path to the folder where the files are stored.
    - classes: A list of the classes we want to include in our dataset.
    - balanced: If balanced is set to True, then the number of examples for each class is balanced.
    Returns:
    - A randomized list of paths to files
    - With its corresponding labels in categorical format
    """
    files = []
    labels = []

    if balanced:
        files_by_category = defaultdict(list)
        min_examples = float('Inf')
        for category in classes:
            files_from_category = [file for file in dataset_path.joinpath(category).iterdir()]
            random.shuffle(files_from_category)
            files_by_category[category].extend(files_from_category)
            num_examples = len(files_by_category[category])
            if num_examples < min_examples:
                min_examples = num_examples
        for index_cat, category in enumerate(classes):
            files.extend(files_by_category[category][:min_examples])
            labels.extend([index_cat]*min_examples)
    else:
        for index_cat, category in enumerate(classes):
            files_from_category = [file for file in dataset_path.joinpath(category).iterdir()]
            files.extend(files_from_category)
            labels.extend([index_cat]*len(files_from_category))

    combined = list(zip(files, labels))
    random.shuffle(combined)
    files[:], labels[:] = zip(*combined)

    return files, to_categorical(labels, len(classes))

def serve_files(files: List[str], labels: List[int], quantity: int) -> Dataset_Tuple:
    """Serve example paths and labels using yield, from the given lists.
    Arguments:
    - files: list of paths to examples
    - labels: list of labels (integer format)
    - quantity: to be served, if quantity is set to 0, yields remaining elements
    Retuns:
    - tuple of lists (files labels) containing the specified quantity.
    """
    first = 0
    while first + quantity < len(labels):
        yield (load_dataset(files[first:first + quantity]), labels[first:first + quantity])
        first += quantity
    yield (files[first:], labels[first:])

def load_dataset(x_files: List[Path]) -> np.ndarray:
    """Load the given dataset files in memory
    """
    x_data = np.array([np.load(file_name).astype('float32') / 1023 for file_name in x_files])
    return x_data
