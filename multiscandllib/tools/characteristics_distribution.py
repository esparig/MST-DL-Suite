import os
import sys
import json
import glob
import matplotlib.pyplot as plt
from pprint import pprint

def print_json(jfile, characteristic):
    data = json.load(jfile)
    pprint(data[characteristic])

def main():
    num_args = len(sys.argv)
    if num_args < 3:
        print("USAGE: python characteristics_distribution.py <dir_path> <characteristic>")
        return

    dir_path = sys.argv[1]
    output_path = os.path.join(dir_path, "histograms")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(2, num_args):
        characteristic = sys.argv[i]
        hist_data = []
        if os.path.isdir(dir_path):
            for fn in glob.glob(os.path.join(dir_path, "*.json")):
                #print(fn)
                with open(fn) as jfile:
                    data = json.load(jfile)
                    val = data[characteristic]
                    if val != None:
                        hist_data.append(val)
        else:
            with open(dir_path) as jfile:
                print_json(jfile)

        n, bins, patches = plt.hist(hist_data)
        plt.title(characteristic.upper())
        plt.xlabel("score S30")
        plt.ylabel("#objects")
        plt.savefig(os.path.join(output_path, characteristic +".png"))
        plt.show()


if __name__ == "__main__":
    main()
