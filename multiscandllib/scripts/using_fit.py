"""Script: training using fit function of keras.
"""
import sys

sys.path.append('..')
sys.path.append('.')

from src.read_arguments import Argument, read_arguments
from src.get_dataset import get_dataset
from src.custom_model import get_model
from src.data_generator import DataGenerator
from src.plot_graphics import plot_performance_graphics, plot_confusion_matrix

def main():
    """Main function:
    - read_arguments
    - get_dataset
    - get_model
    - train
    - evaluate
    """
    arguments = read_arguments()
    