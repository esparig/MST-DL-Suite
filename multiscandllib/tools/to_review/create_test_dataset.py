import os
import pandas as pd
import glob
import shutil

def create_test_dataset(ds_path, dst_dir):
    num_examples = 12
    views = [12]
    category = ['agostadograve', 'agostadoleve', 'granizo', 'molestadograve', 'molestadoleve', 'molino', 'morada', 'picadodemosca', 'primera']

    df = pd.read_csv(os.path.join(ds_path, "examples_detail3.csv"))
    if not os.path.exists(os.path.join(ds_path, dst_dir)):
        os.makedirs(os.path.join(ds_path, dst_dir))

    for c in category:
        print(c)
        if not os.path.exists(os.path.join(ds_path, dst_dir, c)):
            os.makedirs(os.path.join(ds_path, dst_dir, c))
        for v in views:
            print(v)
            for i in range(num_examples):
                id = df[df.category==c][df.nviews==v].values[i][1]
                print(id)
                shutil.copy(os.path.join(ds_path, c, id+"INFO.json"), os.path.join(ds_path, dst_dir, c))
                for j in range(0, 12, 2):
                    shutil.copy(os.path.join(ds_path, c, id+str(j)+"_RGB.png"), os.path.join(ds_path, dst_dir, c))
                    shutil.copy(os.path.join(ds_path, c, id+str(j+1)+"_I.png"), os.path.join(ds_path, dst_dir, c))
                #for file in glob.glob(r''+os.path.join(ds_path, c, id)+"*"):
                #    shutil.copy(file, os.path.join(ds_path, dst_dir, c))

ds_path = os.path.join("D:\\", "olives_multiscan_s30_201811_9categories", "single_pass_test")
create_test_dataset(ds_path, "dataset_miguel")
