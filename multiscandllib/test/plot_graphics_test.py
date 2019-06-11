"""Test plot_graphics.py
"""
import unittest
from multiscandllib.src.plot_graphics import plot_performance_graphics, plot_confusion_matrix


class TestGetDataset(unittest.TestCase):
    """Test Class for plot_graphics.py
    """

    def setUp(self):
        self.history = {'acc':[0.5, 0.55, 0.75], 'val_acc':[0.2, 0.15, 0.3],
                        'loss':[0.2, 0.2, 0.2], 'val_loss':[0.3, 0.4, 0.4]}
