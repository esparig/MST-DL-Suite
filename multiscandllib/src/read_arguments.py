"""
.. module:: read_arguments
    :synopsis: read arguments from CLI
 
.. moduleauthor:: E. Parcero
"""
import os
import argparse
from enum import Enum
from pathlib import Path

class Argument(Enum):
    """Enum class to specify the possible fields in the arguments.
    """
    PROG = 0
    DATASET = 1
    OUTPUT = 2
    BATCH_SIZE = 3
    EPOCHS = 4
    CLASSES = 5
    EXAMPLES = 6
    LR = 7

def read_arguments():
    """Read input arguments::

    --dataset <path to the dataset>
    --output <path to the output folder> #The folder is created if not exists
    --batch_size <desired batch size, typically 32 or 64>
    --epochs <num of epochs to perform>
    --classes [Optional, list of classes] # default: folder names
    --examples <num of examples to employ>
    --lr <learning rate>

    Return:
        A dictionary with the arguments.
    """
    arguments = {}
    # Argument Parser
    parser = argparse.ArgumentParser(description='Example DL training.')
    parser.add_argument('--dataset', metavar='Dataset Path', type=str, required=True,
                        help='a string with the dataset path')
    parser.add_argument('--output', metavar='Output Path', type=str, required=True,
                        help='a string with the output path')
    parser.add_argument('--batch_size', metavar='Batch size', type=int, required=True,
                        help='an integer for the batch size')
    parser.add_argument('--epochs', metavar='Number of Epochs', type=int, required=True,
                        help='an integer for the numer of epochs')
    parser.add_argument('--classes', metavar='Selection of classes in the dataset', nargs='+',
                        help='classes without quotes, separated by a white space')
    parser.add_argument('--examples', metavar='Number of examples', type=int, required=False,
                        help='an integer for the number of examples for training')
    parser.add_argument('--lr', metavar='Learning rate', type=float, required=True,
                        help='a float value for the initial Learning rate used in SGD optimizer')
    args = parser.parse_args()

    print("Executing:", parser.prog)
    print("----------------------------------------")
    print("Dataset:", args.dataset)
    print("Output path:", args.output)
    print("Batch Size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Examples:", args.examples)
    print("Learning rate:", args.lr)


    # Get current script name
    output_folder = args.output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set dataset path
    dataset_path = Path(args.dataset)

    # Set classes
    if args.classes:
        classes = args.classes
    else:
        classes = [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]
    print("Classes:", classes)

    arguments[Argument.PROG] = parser.prog
    arguments[Argument.DATASET] = dataset_path
    arguments[Argument.OUTPUT] = Path(args.output)
    arguments[Argument.BATCH_SIZE] = args.batch_size
    arguments[Argument.EPOCHS] = args.epochs
    arguments[Argument.CLASSES] = classes
    arguments[Argument.EXAMPLES] = args.examples
    arguments[Argument.LR] = args.lr

    return arguments
