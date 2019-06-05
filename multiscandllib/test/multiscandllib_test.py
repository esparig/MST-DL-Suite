"""Try to import my modules
"""
import unittest

import sys
from wheel.signatures import assertTrue
sys.path.append('..')

from src.get_dataset import get_dataset
from src.custom_model import get_model
from src.data_generator import DataGenerator
from src.plot_graphics import plot_performance_graphics

class TestMultiscanDLLib(unittest.TestCase):
    """Test Class for MultiscanDLLib
    """

    def test_imports(self):
        """Test imports.
        """
        self.assertTrue(True)
