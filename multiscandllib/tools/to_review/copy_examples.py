import sys
import os
import glob
from shutil import copyfile

#copyfile(src, dst)

def copy_category_examples (frompath, category, topath, flag_cpy):
    example_list = glob.glob(os.path.join(frompath, "*"+category, "*", "O*"))
    #example_list = glob.glob(os.path.join(frompath, "*"+category, "O*"))
    directories = set()
    i = None
    for i, e in enumerate(example_list):
        dst = os.path.join(topath, os.path.basename(e))
        directories.add(os.path.dirname(e))
        if flag_cpy:
            copyfile (e, dst)
        else:
            print (e, "->", dst)
    if i is not None:
        print( i+1, "copied files")
    else:
        print("String not found")
    for d in sorted(directories):
        print(d)

def main():
    if len(sys.argv) < 5:
        print("USAGE: python copy_examples.py <from_path> <category> <to_path> <-v | -c>")
        return
    frompath, category, topath = sys.argv[1], sys.argv[2], sys.argv[3]
    flag_cpy = False
    if sys.argv[4] == "-c":
        flag_cpy = True

    copy_category_examples(frompath, category, topath, flag_cpy)

if __name__ == "__main__":
    main()
