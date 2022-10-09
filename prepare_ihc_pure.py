import pandas as pd
import os
import numpy as np
import random
import nlpaug.augmenter.word as naw
import argparse

np.random.seed(0)
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default="dataset/implicit-hate-corpus",type=str, help='Enter dataset')
    args = parser.parse_args()

    # load implicit hate corpus dataset
    stg1_posts = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg1_posts.tsv"), delimiter='\t', header=0)
    stg1_ids = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg1.tsv"), delimiter='\t', header=0)

    stg2_posts = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg2_posts.tsv"), delimiter='\t', header=0)
    stg2_ids = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg2.tsv"), delimiter='\t', header=0)

    stg3_posts = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg3_posts.tsv"), delimiter='\t', header=0)
    stg3_posts = stg3_posts.dropna(axis=0).reset_index(drop=True) # there exists a missing value in stg3_posts

    stg3_ids = pd.read_csv(os.path.join(args.load_dir, "implicit_hate_v1_stg3.tsv"), delimiter='\t', header=0)
    stg3_ids = stg3_ids.dropna(axis=0).reset_index(drop=True) # there exists a missing value in stg3_ids

    # merge ids and posts of each stage
    stg1_total = pd.merge(left=stg1_ids, right=stg1_posts, left_index=True, right_index=True, how="inner")
    stg1_total = stg1_total.drop("class_y", axis=1)
    stg1_total.rename(columns = {"class_x": "class"}, inplace=True)

    stg2_total = pd.merge(left=stg2_ids, right=stg2_posts, left_index=True, right_index=True, how="inner")
    stg2_total = stg2_total.drop(["implicit_class_x", "extra_implicit_class_x", "implicit_class_y", "extra_implicit_class_y"], axis=1)

    stg3_total = pd.merge(left=stg3_ids, right=stg3_posts, left_index=True, right_index=True, how="inner")
    stg3_total = stg3_total.drop(['target_x', 'target_y', 'implied_statement_y'], axis=1)
    stg3_total.rename(columns = {"implied_statement_x": "implied_statement"}, inplace=True)

    # construct pure implicit hate
    mask_implicit_total = stg1_total['class'] == 'implicit_hate'
    stg1_implicit_total = stg1_total.loc[mask_implicit_total,:]

    stg1_implicit_stg2_inner_total = pd.merge(left=stg1_implicit_total, right=stg2_total, how="inner", on="ID")
    stg1_implicit_stg2_stg3_inner_total = pd.merge(left=stg1_implicit_stg2_inner_total, right=stg3_total, how="inner", on="ID")
    pure_implicit_total = stg1_implicit_stg2_stg3_inner_total
    pure_implicit_total = pure_implicit_total.drop(["post_x", "post_y"], axis=1)

    # construct pure not hate
    mask_not_hate_total = stg1_total['class'] == 'not_hate'
    stg1_not_hate_total = stg1_total.loc[mask_not_hate_total,:]

    stg2_stg3_outer_total = pd.merge(left=stg2_total, right=stg3_total, how="outer", on="ID")

    pure_not_hate_mask_total = (stg1_not_hate_total['ID'].isin(stg2_stg3_outer_total['ID']) == False)
    pure_not_hate_total = stg1_not_hate_total.loc[pure_not_hate_mask_total, :]

    # construct final pure set
    pure_set = pd.concat([pure_implicit_total, pure_not_hate_total], join='outer')
    pure_set = pure_set.reset_index(drop=True)

    # split to train / valid / test set
    train, valid, test = np.split(pure_set.sample(frac=1, random_state=42), [int(.6*len(pure_set)), int(.8*len(pure_set))])
    # print(len(train)) # 11199
    # print(len(valid)) # 3733
    # print(len(test)) # 3734

    # save train / valid / test set
    os.makedirs("dataset/ihc_pure", exist_ok=True)
    train.to_csv(os.path.join("dataset/ihc_pure", "train.tsv"), sep="\t", index=False)
    valid.to_csv(os.path.join("dataset/ihc_pure", "valid.tsv"), sep="\t", index=False)
    test.to_csv(os.path.join("dataset/ihc_pure", "test.tsv"), sep="\t", index=False)

    # for train set, we include augmented version of posts
    train = pd.read_csv(os.path.join("dataset/ihc_pure", "train.tsv"), sep='\t')
    aug = naw.SynonymAug(aug_src='wordnet')
    train['aug_sent1_of_post'] = pd.Series(dtype="object")
    train['aug_sent2_of_post'] = pd.Series(dtype="object")

    for i,one_post in enumerate(train["post"]):
        train['aug_sent1_of_post'][i] = aug.augment(one_post)
        train['aug_sent2_of_post'][i] = aug.augment(one_post)

    # save train set with augmented version of posts
    train.to_csv(os.path.join("dataset/ihc_pure", "train.tsv"), sep="\t", index=False)