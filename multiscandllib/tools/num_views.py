import sys
import os
import glob
import json
import re
import csv
import pandas as pd
from collections import Counter

def check_nfiles(dir_path, category, id, views):
    errors = []
    seq_views = []
    with open(os.path.join(dir_path, category, id+"INFO.json")) as jfile:
        data = json.load(jfile)
        val = data["views"]
        for i in range(0, views):
            if "I" in val[i]:
                seq_views.append("NIR")
                if not os.path.exists(os.path.join(dir_path, category, id+str(i)+"_I.png")):
                    errors.append(i)
            elif "Color" in val[i]:
                seq_views.append("Color")
                if not os.path.exists(os.path.join(dir_path, category, id+str(i)+"_H.png")) \
                or not os.path.exists(os.path.join(dir_path, category, id+str(i)+"_S.png")) \
                or not os.path.exists(os.path.join(dir_path, category, id+str(i)+"_I.png")):
                    errors.append(i)
            else:
                seq_views.append("Error")
                errors.append(i)

    if len(errors) > 0:
        return False, errors, seq_views
    return True, errors, seq_views

def create_dataframe(dir_path):
    ## TODO
    pass

def write_samples_csv(dir_path):
    report = [["category", "id", "nviews", "type_view", "files_ok", "error_view"]]
    for category in [ folder for folder in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, folder)) ]:
        #print (category)
        for json_list in [ glob.glob(os.path.join(dir_path, category, "*.json")) ]:
            for json_file in json_list:
                m = re.search(r'ObjImg.*_', json_file)
                if m is not None:
                    id = m.group()
                    #print (id)
                    with open(json_file) as jfile:
                        data = json.load(jfile)
                        val = int(data["Num. of Views"])
                        files_ok, error_view, seq_views = check_nfiles(dir_path, category, id, val)
                        #print(files_msg)
                        report.append([category, id, val, seq_views, files_ok, error_view])

    with open(os.path.join(dir_path, 'examples_detail2.csv'), 'w', newline='') as report_file:
        writer = csv.writer(report_file)
        writer.writerows(report)

def read_samples_csv(dir_path):
    samples_dataframe = pd.read_csv(os.path.join(dir_path, 'examples_detail2.csv'))
    #print(samples_dataframe.head(5))
    df_true = samples_dataframe.query('files_ok == True')
    print(df_true.head(5))
    df_false = samples_dataframe.query('files_ok == False')
    print(df_false.head(5))
    print("Ok:", df_true.shape[0], "Fail:", df_false.shape[0])

def main():
    num_args = len(sys.argv)
    if num_args < 2:
        print("USAGE: python num_views.py <dir_path>")
        return

    dir_path = sys.argv[1]

    write_samples_csv(dir_path)

    read_samples_csv(dir_path)



if __name__ == "__main__":
    main()
