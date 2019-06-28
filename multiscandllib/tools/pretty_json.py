import os
import sys
import json
import glob
from pprint import pprint

def print_json(jfile):
    data = json.load(jfile)
    pprint(data)

def main():
    if len(sys.argv) < 2:
        print("USAGE: python pretty_json.py <dir_path>")
        return

    dir_path = sys.argv[1]
    if os.path.isdir(dir_path):
        for fn in glob.glob(os.path.join(dir_path, "*.json")):
            print(fn)
            with open(fn) as jfile:
                print_json(jfile)
    else:
        with open(dir_path) as jfile:
            print_json(jfile)

if __name__ == "__main__":
    main()
