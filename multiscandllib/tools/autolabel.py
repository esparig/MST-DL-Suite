import os
import json
import glob
from shutil import copyfile
from pprint import pprint


def autolabel():
    if not os.path.exists("Category0"):
        os.mkdir("Category0")
    if not os.path.exists("Category1"):
        os.mkdir("Category1")
    if not os.path.exists("Category2"):
        os.mkdir("Category2")
    for fn in glob.glob("*.bmp"):
        fn_json = fn.split("RGB.bmp")[0]+"INFO.json"
        print(fn_json)
        with open(fn_json) as o_json:
                data = json.load(o_json)
                #print(data['Category'])
                if data['Category'] == 0:
                    copyfile(fn, os.path.join("Category0", fn))
                if data['Category'] == 1:
                    copyfile(fn, os.path.join("Category1", fn))
                if data['Category'] == 2:
                    copyfile(fn, os.path.join("Category2", fn))
                #pprint(data)

autolabel()
