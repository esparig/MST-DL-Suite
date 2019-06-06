"""Try to import my modules
"""
import unittest

from multiscandllib.src.get_dataset import get_dataset
from multiscandllib.src.custom_model import get_model
from multiscandllib.src.data_generator import DataGenerator
from multiscandllib.src.plot_graphics import plot_performance_graphics

class TestMultiscanDLLib(unittest.TestCase):
    """Test Class for MultiscanDLLib
    """

    def test_imports(self):
        """Test imports.
        """
        self.assertTrue(True)
