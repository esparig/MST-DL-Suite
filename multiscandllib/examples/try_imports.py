"""Try to import my modules
"""
import sys
sys.path.append('..')

from src.get_dataset import get_dataset
from src.custom_model import get_model
from src.data_generator import DataGenerator
from src.plot_graphics import plot_performance_graphics

print("SUCCESS!!")
