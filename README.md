# Generalizable Implicit Hate Speech Detection using Contrastive Learning
This repository contains the code for the paper *Generalizable Implicit Hate Speech Detection using Contrastive Learning*.

## Requirements
You can download the requirements by:
```
pip install -r requirements.txt
```

## Prepare Dataset
First, download [Implicit Hate Corpus](https://github.com/SALT-NLP/implicit-hate) (IHC), [Social Bias Inference Corpus](https://maartensap.com/social-bias-frames) (SBIC), and [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset).
Then, run the following code to prepare the dataset. This code splits dataset and prepare augmented version of posts and implications.
```
python prepare_ihc_pure.py --load_dir [DIRECTORY_TO_IHC]
python prepare_sbic.py --load_dir [DIRECTORY_TO_SBIC]
python prepare_dynahate.py --load_dir [DIRECTORY_TO_DYNAHATE]
```

## Data Preprocess
You can preprocess the dataset using data_preprocess.py. 

To get preprocessed ihc dataset for *AugCon* (CE+*AugCon*):
```
python data_preprocess.py -d ihc_pure --aug
```

To get preprocessed ihc dataset for *ImpCon* (CE+*ImpCon*):
```
python data_preprocess.py -d ihc_pure_imp --aug
```

To get preprocessed sbic dataset for *AugCon* (CE+*AugCon*):
```
python data_preprocess.py -d sbic --aug
```

To get preprocessed sbic dataset for *ImpCon* (CE+*ImpCon*):
```
python data_preprocess.py -d sbic_imp --aug
```

## Train
You can train a model by:
```
python train.py
```
The example train_config.py file is for training bert-base-uncased model with CE+*ImpCon*.
You can also modify config.py to train a model with different training objectives or evaluate a model.
### Train Configs
- To train a model on ihc with CE+*AugCon*:
    ```
    dataset = ["ihc_pure"]
    ...
    w_aug = True
    w_double = False
    w_separate = False
    w_sup = False
    ...
    ```
- To train a model on sbic with CE+*AugCon*:
    ```
    dataset = ["sbic"]
    ...
    w_aug = True
    w_double = False
    w_separate = False
    w_sup = False
    ...
    ```
- To train a model on ihc with CE+*ImpCon*:
    ```
    dataset = ["ihc_pure_imp"]
    ...
    w_aug = True
    w_double = False
    w_separate = False
    w_sup = False
    ...
    ```
- To train a model on sbic with CE+*ImpCon*:
    ```
    dataset = ["sbic_imp"]
    ...
    w_aug = True
    w_double = False
    w_separate = False
    w_sup = False
    ...
    ```
- To train hatebert model, first download [hatebert](https://osf.io/tbd58/?view_onlycb79b3228d4248ddb875eb1803525ad8). You can download retrained_model/hate_bert.zip and unzip it in the root directory. Then, 
    ```
    ...
    model_type = "hatebert"
    ...
    ```
## Evaluation
You can evaluate the saved model by:
```
python eval.py
```
### Evaluation Configs
- Before executing the code, change the load_dir in eval_config.py:
    ```
    ...
    load_dir = [DIRECTORY_TO_SAVED_MODEL]
    ...
    ```
    The evaluation results will be saved in load_dir.
- You can set datasets where the model is evaluated on:
    ```
    ...
    dataset = ["ihc_pure", "sbic", "dynahate"] # dataset for evaluation
    ...
    ```
## Acknowledgement
Our code is based on the code from https://github.com/varsha33/LCL_loss.

Also, prepare_sbic.py is based on the code from https://github.com/allenai/feb.