import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default="dataset/DynaHate",type=str, help='Enter dataset')
    args = parser.parse_args()

    dynahate_dataset = pd.read_csv(os.path.join(args.load_dir, "DynaHate_v0.2.2.csv"), delimiter=',', header=0)
    dynahate_dataset = dynahate_dataset.drop(dynahate_dataset.columns[0], axis=1)

    mask_train = dynahate_dataset['split'] == 'train'
    dynahate_train = dynahate_dataset.loc[mask_train,:]

    mask_dev = dynahate_dataset['split'] == 'dev'
    dynahate_dev = dynahate_dataset.loc[mask_dev,:]

    mask_test = dynahate_dataset['split'] == 'test'
    dynahate_test = dynahate_dataset.loc[mask_test,:]

    os.makedirs("dataset/DynaHate", exist_ok=True)
    dynahate_train.to_csv(os.path.join("dataset/DynaHate", "train.csv"), sep=",", index=False)
    dynahate_dev.to_csv(os.path.join("dataset/DynaHate", "dev.csv"), sep=",", index=False)
    dynahate_test.to_csv(os.path.join("dataset/DynaHate", "test.csv"), sep=",", index=False)