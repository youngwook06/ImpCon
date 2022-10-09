import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch import nn

import eval_config as train_config
from dataset_impcon import get_dataloader
from util import iter_product
from sklearn.metrics import f1_score
import loss_impcon as loss
from model import primary_encoder_v2_no_pooler_for_con

from transformers import AdamW,get_linear_schedule_with_warmup, BertForSequenceClassification 

from tqdm import tqdm

# Credits https://github.com/varsha33/LCL_loss
def test(test_loader,model_main,log):
    model_main.eval()
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}

    total_feature = []
    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            if "ihc" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "dynahate" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "sbic" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            else:
                text_name = "cause"
                label_name = "emotion"
                raise NotImplementedError

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
            pred_1 = model_main(last_layer_hidden_states)

            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature_1.data.detach().cpu().tolist())
            total_pred_prob_1.extend(pred_1.data.detach().cpu().tolist())

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    save_pred["feature"] = total_feature
    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc,f1_score_1,save_pred

##################################################################################################
def cl_test(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True #
    torch.backends.cudnn.benchmark = False #



    print("#######################start run#######################")
    print("log:", log)

    _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None)


    model_main = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    
    #################################################################
    # load model
    model_main.load_state_dict(torch.load(os.path.join(log.param.load_dir, "model.pt")))
    print(f"model is loaded from {log.param.load_dir}")
    
    model_main.eval()
    if torch.cuda.is_available():
        model_main.cuda()
    ###################################################################
    
    val_acc_1,val_f1_1,val_save_pred = test(valid_data,model_main,log)
    test_acc_1,test_f1_1,test_save_pred = test(test_data,model_main,log)

    print("Model 1")
    print(f'Valid Accuracy: {val_acc_1:.2f} Valid F1: {val_f1_1["macro"]:.2f}')
    print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]:.2f}')

    log.valid_f1_score_1 = val_f1_1
    log.test_f1_score_1 = test_f1_1
    log.valid_accuracy_1 = val_acc_1
    log.test_accuracy_1 = test_acc_1

    if log.param.dataset == "dynahate":
        with open(os.path.join(log.param.load_dir, "dynahate_test_log.json"), 'w') as fp:
            json.dump(dict(log), fp,indent=4)
    elif "sbic" in log.param.dataset:
        with open(os.path.join(log.param.load_dir, "sbic_test_log.json"), 'w') as fp:
            json.dump(dict(log), fp,indent=4)
    elif "ihc" in log.param.dataset:
        with open(os.path.join(log.param.load_dir, "ihc_test_log.json"), 'w') as fp:
            json.dump(dict(log), fp,indent=4)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 2
        
        assert log.param.load_dir is not None, "to load a model, log.param.load_dir should be given!!"
        cl_test(log)
