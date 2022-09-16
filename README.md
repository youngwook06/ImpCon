# Generalizable Implicit Hate Speech Detection using Contrastive Learning
This repository contains the code for the paper *Generalizable Implicit Hate Speech Detection using Contrastive Learning*.

## Data Preprocess
You can preprocess the dataset using data_preprocess.py. To get preprocessed ihc dataset for *ImpCon*:
```
python data_preprocess.py -d ihc_pure_imp --aug
```

## Train
You can train a model with *ImpCon* using the example config.py file:
```
python train.py
```
You can also modify config.py to train a model with different training objectives or evaluate a model.

## Acknoweldgement
Our code is based on the code from https://github.com/varsha33/LCL_loss. 