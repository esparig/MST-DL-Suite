"""Create and save numpy files from images to be use as a input dataset for deep learning.
H, S, I images -> 1 numpy file. (3 layers of depth)
"""
import os
import argparse
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

class Argument(Enum):
    """Enum class to specify the possible fields in the arguments
    """
    PROG = 0
    INPUT = 1
    OUTPUT = 2
    CLASSES = 3

def get_arguments():
    """Read input arguments:

    --input <path to the input folder>
    --output <path to the output folder> The folder is created if not exists
    --classes [Optional, list of classes] If it isn't specified, all the classes
    inside the dataset path are used

    Returns a dict with the arguments.
    """
    arguments = {}
    # Argument Parser
    parser = argparse.ArgumentParser(description='Create numpy files from image dataset')
    parser.add_argument('--input', metavar='Input Path', type=str, required=True,
                        help='a string with the path to the input folder')
    parser.add_argument('--output', metavar='Output Path', type=str, required=True,
                        help='a string with the path to the output folder')
    parser.add_argument('--classes', metavar='Selection of classes in the dataset', nargs='+',
                        help='classes without quotes, separated by a white space')
    args = parser.parse_args()

    # Set dataset path
    input_folder = Path(args.input)

    # Get current script name
    output_folder = Path(args.output)
    if not output_folder.exists():
        os.makedirs(output_folder)

    # Set classes
    if args.classes:
        classes = args.classes
    else:
        classes = [folder.name for folder in input_folder.iterdir() if folder.is_dir()]

    print("Executing:", parser.prog)
    print("----------------------------------------")
    print("Input path:", args.input)
    print("Output path:", args.output)
    print("Classes:", classes)

    arguments[Argument.PROG] = parser.prog
    arguments[Argument.INPUT] = input_folder
    arguments[Argument.OUTPUT] = output_folder
    arguments[Argument.CLASSES] = classes

    return arguments


def gen_dataset(in_folder: Path, output_folder: Path, classes: List[str], write: bool):
    """Saves the numpy files generated from existing HSI layers in the in_folder folder
    Inputs:
    - in_folder: input folder
    - output_folder: output_folder folder
    - classes: selection of class folders to use
    - write: if True saves to disk (into output_folder folder), if False print files to be generated
    """
    for current_class in classes:
        if not Path(output_folder/current_class).exists():
            os.makedirs(Path(output_folder/current_class))
        print(current_class, "\n-------------")
        for file in tqdm(list(Path(in_folder/current_class).glob('*.jpg'))):
            try:
                img = io.imread(file)
                img2 = resize(img, (200, 200, 3))
                npy_name = output_folder.joinpath(current_class, file.parts[6][:-4]+'.npy')
                np.save(npy_name, img2)
            except IOError:
                print("Could not save image", file)
            '''
            if np.random.random_sample() < 0.001:
                my_npy = np.load(npy_name)
                plt.imshow(my_npy)
                plt.show()
            '''

def main():
    """Main function:
    1. get arguments provided via CLI
    2. generate new dataset
    """

    args = get_arguments()
    gen_dataset(args[Argument.INPUT], args[Argument.OUTPUT], args[Argument.CLASSES], True)

if __name__ == "__main__":
    main()
