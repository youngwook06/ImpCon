import pandas as pd 
import spacy 
import argparse
from tqdm import trange
import numpy as np
import os
import random
import nlpaug.augmenter.word as naw
# import jsonlines 

np.random.seed(0)
random.seed(0)

# credits: https://github.com/allenai/feb
def aggregate_sbic_annotations(split, dataset_path):
    '''
    In the original SBIC csv file, one post occurs multiple times with annotations from different workers. 
    Here, for each post, we aggregate its annotations into a single row (for eval instances) or make multiple train instances. 
    '''
    df = pd.read_csv(os.path.join(dataset_path, f"SBIC.v2.{split}.csv"))
    columns = ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
    aggregated_data = []
    visited_posts = []
    for i in trange(len(df["targetStereotype"])):
        post = df.loc[i, "post"]
        if post in visited_posts:
            continue
        visited_posts.append(post)

        # A post is offensive if at least half of the annotators say it is. 
        offensiveYN_frac = sum(df.loc[df["post"]==post]["offensiveYN"]) / float(len(df.loc[df["post"]==post]["offensiveYN"]))
        offensiveYN_label = 1.0 if offensiveYN_frac >= 0.5 else 0.0

        # A post targets a demographic group if at least half of the annotators say it does.
        whoTarget_frac = sum(df.loc[df["post"]==post]["whoTarget"]) / float(len(df.loc[df["post"]==post]["whoTarget"]))
        whoTarget_label = 1.0 if whoTarget_frac >= 0.5 else 0.0

        targetMinority_label = None 
        targetStereotype_label = None
        
        if whoTarget_label == 1.0: # The post targets an identity group; only such posts have annotations of stereotypes of the group that are referenced or implied
            minorities = df.loc[df["post"]==post]["targetMinority"]
            stereotypes = df.loc[df["post"]==post]["targetStereotype"]

            if split in ['dev', 'tst']: # For evaluation, we combine all implied statements into a single string separated by [SEP] 
                targetMinority_labels = []
                targetStereotype_labels = []
                for m, s in zip(minorities, stereotypes):
                    if not pd.isna(s):
                        targetMinority_labels.append(m)
                        targetStereotype_labels.append(s)
                targetMinority_label = ' [SEP] '.join(targetMinority_labels)
                targetStereotype_label = ' [SEP] '.join(targetStereotype_labels)
                aggregated_data.append([post, offensiveYN_label, whoTarget_label, targetMinority_label, targetStereotype_label])
            else: # For training, each implied statement leads to an individual training instance
                temp_aggregated_data = []
                for m, s in zip(minorities, stereotypes):
                    if not pd.isna(s):
                        temp_aggregated_data.append([post, offensiveYN_label, whoTarget_label, m, s])
                if len(temp_aggregated_data) >0:
                    one_data_for_one_post = random.choice(temp_aggregated_data)
                else:
                    one_data_for_one_post = [post, offensiveYN_label, whoTarget_label, m, s]
                aggregated_data.append(one_data_for_one_post)
                        
        else: 
            aggregated_data.append([post, offensiveYN_label, whoTarget_label, targetMinority_label, targetStereotype_label])
    df_new = pd.DataFrame(aggregated_data, columns=columns) 
    return df_new

# credits: https://github.com/allenai/feb
def turn_implied_statements_to_explanations(split, df):
    '''
    This function implements a set of rules to transform annotations of which identity-based group is targeted and what stereotypes of this group are referenced or implied into a single, coherent sentence (explanation).
    For example:
    `targetMinority` == "women"
    `targetStereotype` == "can't drive"
    return: "this posts implies that women can't drive." 

    For attacks on individuals, it will return "this post is a personal attack". 

    For posts that are not offensive, it will return "this post does not imply anything offensive"
    '''
    if df is None:  
        raise NotImplementedError

    df['selectedStereotype'] = pd.Series(dtype="object")

    group_attack_no_implied_statement = 0
    personal_attack = 0
    not_offensive = 0
    group_offensive = 0
    offensive_na_whotarget = 0

    for i in trange(len(df["targetStereotype"])):
        offensive_label = df.loc[i,"offensiveLABEL"]

        if offensive_label == 'offensive' and (pd.isna(df.loc[i, "whoTarget"]) or df.loc[i, "whoTarget"]==''):
            offensive_na_whotarget+=1
            continue

        if offensive_label == 'offensive' and df.loc[i,"whoTarget"] == 1.0: # only posts that target a group have annotations of implied statements 
            if pd.isna(df.loc[i, "targetStereotype"]) or df.loc[i, "targetStereotype"]=='': 
                group_attack_no_implied_statement += 1 
                continue
            inferences = df.loc[i, "targetStereotype"].split(' [SEP] ')
            target_minorities = df.loc[i, "targetMinority"].split(' [SEP] ')
            
            new_inferences = []
            for target_minority, inference in zip(target_minorities,inferences): 
                inference_annotations = [{'token': token.text, 'pos': token.pos_, 'tag': token.tag_} for token in nlp(inference)]
                first_word = inference_annotations[0]
                changed = False 

                if first_word['token'] == 'claims' and first_word['tag'] not in ['VBD', 'VBZ']: 
                    new_inferences.append(f'this post {inference}')
                    changed = True

                if first_word['tag'] in ['VBD', 'VBZ']: # verb, 3rd person singular present or simple past tense (e.g, said, calls, makes, uses, implies, marginalized, advocates, encourages, trivializes, refers) #ywywyw TODO is 가 들어가는 경우에 이상한 것 같은데 check
                    new_inferences.append(f'this post {inference}')
                    changed = True
                
                if (first_word['tag'] == 'VBG' and first_word['token'] not in ['implying', 'calling']) or first_word['token'] == 'being': # gerund or present participle and "being" [infinitive (e.g, using, lying, having, disrespecting, being; BUT NOT implying or calling)]
                    new_inferences.append(f'this post implies that {inference}')
                    changed = True

                if first_word['token'] in ['implying', 'calling']:
                    new_inferences.append(f'this post is {inference}')
                    changed = True
                
                if first_word['token'] == 'rape':
                    new_inferences.append(f'this post implies that {target_minority} {inference}')
                    changed = True
                
                if first_word['tag'] in ['MD', 'VB'] or (first_word['pos'] == 'AUX' and first_word['token'] != 'being'): # verb, modal verb or base form 
                    if (target_minority in ["women", "men"] or "folks" in target_minority or len(target_minority.split(',')) > 1) and first_word['token'] == 'is': # verb does not match targets' number
                        split_inference = ' '.join(inference.split(' ')[1:])
                        new_inferences.append(f'this post implies that {target_minority} are {split_inference}')
                        changed = True
                    else:
                        new_inferences.append(f'this post implies that {target_minority} {inference}')
                        changed = True
                
                if first_word['token'] == "all":
                    new_inferences.append(f'this post implies that {target_minority} are {inference}')
                    changed = True
                
                if not changed: 
                    new_inferences.append(f'this post implies that {inference}')
                group_offensive += 1
            if len(new_inferences) > 1:
                df.loc[i, "selectedStereotype"] = random.choice(new_inferences)
            else: 
                df.loc[i, "selectedStereotype"] = new_inferences[0]

        if offensive_label == 'offensive' and df.loc[i,"whoTarget"] == 0.0:
            personal_attack += 1

        if offensive_label == 'not_offensive':
            not_offensive += 1

    
    print ("---------------------------------------------------")
    print (f"Split: {split}")    
    print (f"offensive_na_whotarget: {offensive_na_whotarget}") # 0
    print (f"Group attack but no implied statement: {group_attack_no_implied_statement}") # 3
    print (f"Personal attacks: {personal_attack}") # 6082
    print (f"Group offensive: {group_offensive}") # 12008
    print (f"Not offensive: {not_offensive}") # 17411
    print ("---------------------------------------------------")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', default="dataset/SBIC.v2",type=str, help='Enter dataset')
    args = parser.parse_args()
    
    # aggregate annotations
    SBIC_train = aggregate_sbic_annotations(split='trn', dataset_path=args.load_dir)
    SBIC_dev = aggregate_sbic_annotations(split='dev', dataset_path=args.load_dir)
    SBIC_test = aggregate_sbic_annotations(split='tst', dataset_path=args.load_dir)
    # print(len(SBIC_train)) # trn : 35504
    # print(len(SBIC_dev)) # dev : 4673
    # print(len(SBIC_test)) # tst : 4698

    SBIC_train['offensiveLABEL'] = np.where(SBIC_train['offensiveYN']>=0.5, 'offensive', 'not_offensive')
    SBIC_dev['offensiveLABEL'] = np.where(SBIC_dev['offensiveYN']>=0.5, 'offensive', 'not_offensive')
    SBIC_test['offensiveLABEL'] = np.where(SBIC_test['offensiveYN']>=0.5, 'offensive', 'not_offensive')

    # save dev / test set
    os.makedirs("dataset/SBIC.v2", exist_ok=True)
    SBIC_dev.to_csv(os.path.join("dataset/SBIC.v2", "dev.csv"), sep=",", index=False)
    SBIC_test.to_csv(os.path.join("dataset/SBIC.v2", "test.csv"), sep=",", index=False)

    # augmented version of posts for train set
    aug = naw.SynonymAug(aug_src='wordnet')
    SBIC_train['aug_sent1_of_post'] = pd.Series(dtype="object")
    SBIC_train['aug_sent2_of_post'] = pd.Series(dtype="object")
    for i,one_post in enumerate(SBIC_train["post"]):
        SBIC_train.loc[i, 'aug_sent1_of_post'] = aug.augment(one_post)
        SBIC_train.loc[i, 'aug_sent2_of_post'] = aug.augment(one_post)
        
    # implication for train set 
    nlp = spacy.load("en_core_web_sm")
    SBIC_train_modified = turn_implied_statements_to_explanations(split='trn', df=SBIC_train)
    SBIC_train_modified.to_csv(os.path.join("dataset/SBIC.v2", "train.csv"), sep=",", index=False)

