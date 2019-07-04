"""Test get_dataset.py
"""
import unittest
from pathlib import Path
from numpy import argmax
from multiscandllib.src.get_dataset import get_dataset, serve_files, get_files


class TestGetDataset(unittest.TestCase):
    """Test Class for get_dataset.py
    """

    def setUp(self):
        self.ds_path = Path("/media/deeplearning/SSD2TB/mini_dataset_3cats")
        self.x, self.lx, self.y, self.ly, self.z, self.lz = get_dataset(
            self.ds_path,percent_train=80,percent_val=10,percent_test=10)
        self.classes = [folder.name for folder in self.ds_path.iterdir() if folder.is_dir()]

    def test_warning(self):
        """Test if warning is showed
        """
        self.assertWarns(Warning,
                         get_dataset,
                         self.ds_path,percent_train=80,percent_val=20,percent_test=20)

    def test_dataset_set_classes(self):
        """Test if selection of classes
        """
        my_classes = ['primera', 'molestadograve']
        samples, labels, _, _, _, _ = get_dataset(
            self.ds_path, percent_train=100, percent_val=0, percent_test=0,
            classes = my_classes)
        for pair in [(sample.parent.name, my_classes[argmax(label)])
                     for sample, label in zip(samples, labels)]:
            self.assertEqual(pair[0], pair[1])

    def test_dataset_is_complete(self):
        """Test if the returned dataset is complete.
        """
        set_from_get_dataset = set([*self.x, *self.y, *self.z])

        set_from_folder = set()
        for category in self.classes:
            for file in self.ds_path.joinpath(category).iterdir():
                set_from_folder.add(file)
        print()
        print("Elements from get_dataset")
        print("-------------------------")
        for elem1 in set_from_get_dataset:
            print(elem1)
        print("Elements from folder")
        print("--------------------")
        for elem2 in set_from_folder:
            print(elem2)
        self.assertEqual(set_from_get_dataset, set_from_folder)

    def test_dataset_independent(self):
        """Test if the returned dataset contains different
        files in training, validation, and test subsets
        """
        set_from_get_dataset = set([*self.x, *self.y, *self.z])
        self.assertTrue(len(self.x)+len(self.y)+len(self.z) == len(set_from_get_dataset))

    def test_dataset_labels(self):
        """Test if the returned dataset is
        correctly labelled
        """
        for pair in [(sample.parent.name, self.classes[argmax(label)])
                     for sample, label in zip(self.y, self.ly)]:
            self.assertEqual(pair[0], pair[1])

    def test_serve_files(self):
        """Test serve files function
        """
        files, labels = get_files(self.ds_path, self.classes, False)
        sf1 = serve_files(files, labels, 10)
        print("Served files")
        print("------------")
        print(type(sf1))
