import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time

import torch
import torch.utils.data
from torch import nn

import config as train_config
from dataset_imp_con import get_dataloader
from util import iter_product
from sklearn.metrics import f1_score
import loss_imp_con as loss
from model import primary_encoder_v2_no_pooler_for_con

from transformers import AdamW,get_linear_schedule_with_warmup, BertForSequenceClassification 

from tqdm import tqdm

# Credits https://github.com/varsha33/LCL_loss
def train(epoch,train_loader,model_main,loss_function,optimizer,lr_scheduler,log):

    model_main.cuda()
    model_main.train()

    total_true,total_pred_1,acc_curve_1 = [],[],[]
    train_loss_1 = 0
    total_epoch_acc_1 = 0
    steps = 0
    start_train_time = time.time()

    if log.param.w_aug:
        if log.param.w_double:
            train_batch_size = log.param.train_batch_size*3
        else:
            train_batch_size = log.param.train_batch_size*2 # only for w_aug
    else:
        train_batch_size = log.param.train_batch_size
    print("train with aug:", log.param.w_aug)
    print("train with double aug:", log.param.w_double)
    print("train with separate double aug:", log.param.w_separate)
    print("loss with sup(using label info):", log.param.w_sup)
    print("len(train_loader):", len(train_loader))
    print("train_batch_size:", train_batch_size)
    if log.param.w_separate:
        assert log.param.w_double, "w_double should be set to True for w_separate=True option"
    
    # for idx,batch in tqdm(enumerate(tqdm(train_loader))):
    for idx,batch in enumerate(train_loader):
        # if "ihc" in log.param.dataset or "sbic" in log.param.dataset:
        if "ihc" in log.param.dataset or "sbic" in log.param.dataset or 'dynahate' in log.param.dataset:
            text_name = "post"
            label_name = "label"
        else:
            raise NotImplementedError

        text = batch[text_name]
        attn = batch[text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()

        if (label.size()[0] is not train_batch_size):# Last batch may have length different than log.param.batch_size
            continue

        if torch.cuda.is_available():
            text = text.cuda()
            attn = attn.cuda()
            label = label.cuda()

        #####################################################################################
        if log.param.w_aug: # text split
            if log.param.w_double:
                if log.param.w_separate:
                    assert log.param.train_batch_size == label.shape[0] // 3
                    assert label.shape[0] % 3 == 0
                    original_label, augmented_label_1, augmented_label_2 = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    only_original_labels = original_label

                    original_text, augmented_text_1, augmented_text_2 = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    original_attn, augmented_attn_1, augmented_attn_2 = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)

                    original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn) 

                    _, augmented_supcon_feature_1_1 = model_main.get_cls_features_ptrnsp(augmented_text_1,augmented_attn_1)
                    _, augmented_supcon_feature_1_2 = model_main.get_cls_features_ptrnsp(augmented_text_2,augmented_attn_2)

                    supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_1], dim=0)
                    supcon_feature_2 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_2], dim=0)

                    assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size

                    pred_1 = model_main(original_last_layer_hidden_states)

                else:
                    assert log.param.train_batch_size == label.shape[0] // 3
                    assert label.shape[0] % 3 == 0
                    original_label, augmented_label_1, augmented_label_2 = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    only_original_labels = original_label

                    original_text, augmented_text_1, augmented_text_2 = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)
                    original_attn, augmented_attn_1, augmented_attn_2 = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size, log.param.train_batch_size], dim=0)

                    original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn)

                    _, augmented_supcon_feature_1_1 = model_main.get_cls_features_ptrnsp(augmented_text_1,augmented_attn_1)
                    _, augmented_supcon_feature_1_2 = model_main.get_cls_features_ptrnsp(augmented_text_2,augmented_attn_2)

                    supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1_1, augmented_supcon_feature_1_2], dim=0)

                    assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size

                    pred_1 = model_main(original_last_layer_hidden_states)

            else:
                assert log.param.train_batch_size == label.shape[0] // 2
                assert label.shape[0] % 2 == 0
                original_label, augmented_label = torch.split(label, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                only_original_labels = original_label

                original_text, augmented_text = torch.split(text, [log.param.train_batch_size, log.param.train_batch_size], dim=0)
                original_attn, augmented_attn = torch.split(attn, [log.param.train_batch_size, log.param.train_batch_size], dim=0)

                original_last_layer_hidden_states, original_supcon_feature_1 = model_main.get_cls_features_ptrnsp(original_text, original_attn) # #v2
                
                _, augmented_supcon_feature_1 = model_main.get_cls_features_ptrnsp(augmented_text,augmented_attn) # #v2

                supcon_feature_1 = torch.cat([original_supcon_feature_1, augmented_supcon_feature_1], dim=0)
                assert original_last_layer_hidden_states.shape[0] == log.param.train_batch_size

                pred_1 = model_main(original_last_layer_hidden_states)

        else:
            if not log.param.debug:
                assert log.param.train_batch_size == label.shape[0]
                only_original_labels = label
                last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
                pred_1 = model_main(last_layer_hidden_states)
            else:
                only_original_labels = label
                pred_1 = model_main(text,attn, output_hidden_states=True,return_dict=True)
                pred_1 = pred_1.logits 


        if log.param.w_aug and log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    raise NotImplementedError
                else:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive_for_double"](supcon_feature_1,label)) # w_aug_w_sup 의 경우는 두개가 같을 가능성이 높을 것 같음. 확인해보고 합치든가 하기
            else:
                loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive"](supcon_feature_1,label))
        elif log.param.w_aug and not log.param.w_sup:
            if log.param.w_double:
                if log.param.w_separate:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((0.5*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](supcon_feature_1)) + ((0.5*(1-loss_function["lambda_loss"]))*loss_function["contrastive"](supcon_feature_2)) # label 안주면 sup 정보 활용안하도록 되어있음. 즉, aug 또는 implication 만 positive 로 봄
                else:
                    loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive_for_double"](supcon_feature_1)) # label 안주면 sup 정보 활용안하도록 되어있음. 즉, aug 또는 implication 만 positive 로 봄
            else:
                loss_1 = (loss_function["lambda_loss"]*loss_function["ce_loss"](pred_1,only_original_labels)) + ((1-loss_function["lambda_loss"])*loss_function["contrastive"](supcon_feature_1)) # label 안주면 sup 정보 활용안하도록 되어있음. 즉, aug 또는 implication 만 positive 로 봄
        else: # log.param.w_aug == False
            loss_1 = loss_function["ce_loss"](pred_1,only_original_labels)


        loss = loss_1
        train_loss_1  += loss_1.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)
        optimizer.step()
        model_main.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print (f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss_1: {loss_1.item():.4f}, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()

        true_list = only_original_labels.data.detach().cpu().tolist()
        total_true.extend(true_list)

        num_corrects_1 = (torch.max(pred_1, 1)[1].view(only_original_labels.size()).data == only_original_labels.data).float().sum()
        pred_list_1 = torch.max(pred_1, 1)[1].view(only_original_labels.size()).data.detach().cpu().tolist()

        total_pred_1.extend(pred_list_1)

        acc_1 = 100.0 * (num_corrects_1/log.param.train_batch_size)
        acc_curve_1.append(acc_1.item())
        total_epoch_acc_1 += acc_1.item()

    print(train_loss_1/len(train_loader))
    print(total_epoch_acc_1/len(train_loader))

    return train_loss_1/len(train_loader),total_epoch_acc_1/len(train_loader),acc_curve_1


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

            if not log.param.debug:
                last_layer_hidden_states, supcon_feature_1 = model_main.get_cls_features_ptrnsp(text,attn) # #v2
                pred_1 = model_main(last_layer_hidden_states)
            else:
                pred_1 = model_main(text,attn, output_hidden_states=True,return_dict=True)
                pred_1 = pred_1.logits

            num_corrects_1 = (torch.max(pred_1, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list_1 = torch.max(pred_1, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            if not log.param.debug:
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

def cl_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    print("#######################start run#######################")
    print("log:", log)
    train_data,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=log.param.w_aug,w_double=log.param.w_double,label_list=None)
    print("len(train_data):", len(train_data)) 

    losses = {"contrastive":loss.SupConLoss(temperature=log.param.temperature),"ce_loss":nn.CrossEntropyLoss(),"lambda_loss":log.param.lambda_loss,"contrastive_for_double":loss.SupConLoss_for_double(temperature=log.param.temperature)}

    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    if not log.param.debug:
        model_main = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type)
    else:
        model_main = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    if 'ihc' in log.param.dataset:
        optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate, eps=1e-8) # eps=1e-8 following latent hatred
        print("For ihc, eps for AdaW optimizer is set to 1e-8 following latent hatred")
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate) # other than ihc, use default eps (which is 1e-6)
        print("eps for AdaW optimizer is set to default (1e-6)")
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    if log.param.run_name != "":
        save_home = "./save/"+str(log.param.SEED)+"/"+log.param.dataset+"/"+log.param.run_name+"/"+log.param.loss_type+"/"+model_run_time+"/"
    else:
        save_home = "./save/"+str(log.param.SEED)+"/"+log.param.dataset+"/"+log.param.loss_type+"/"+model_run_time+"/"

    total_train_acc_curve_1, total_val_acc_curve_1 = [],[]

    for epoch in range(1, log.param.nepoch + 1):

        train_loss_1,train_acc_1,train_acc_curve_1 = train(epoch,train_data,model_main, losses,optimizer,lr_scheduler,log)
        val_acc_1,val_f1_1,val_save_pred = test(valid_data,model_main,log)
        test_acc_1,test_f1_1,test_save_pred = test(test_data,model_main,log)

        total_train_acc_curve_1.extend(train_acc_curve_1)

        print('====> Epoch: {} Train loss_1: {:.4f}'.format(epoch, train_loss_1))

        os.makedirs(save_home,exist_ok=True)
        with open(save_home+"/acc_curve.json", 'w') as fp:
            json.dump({"train_acc_curve_1":total_train_acc_curve_1}, fp,indent=4)

        if epoch == 1:
             best_criterion = 0.0

        ########### best model by val_f1_1["macro"]
        is_best = val_f1_1["macro"] > best_criterion
        best_criterion = max(val_f1_1["macro"],best_criterion)

        print("Model 1")
        print("Best model evaluated by macro f1")
        print(f'Valid Accuracy: {val_acc_1:.2f} Valid F1: {val_f1_1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]:.2f}')


        if is_best:
            print("======> Best epoch <======")
            log.train_loss_1 = train_loss_1
            log.stop_epoch = epoch
            log.valid_f1_score_1 = val_f1_1
            log.test_f1_score_1 = test_f1_1
            log.valid_accuracy_1 = val_acc_1
            log.test_accuracy_1 = test_acc_1
            log.train_accuracy_1 = train_acc_1

            ## load the model
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp,indent=4)
            fp.close()

            ###############################################################################
            # save model
            if log.param.save:
                torch.save(model_main.state_dict(), os.path.join(save_home, 'model.pt'))
                print(f"best model is saved at {os.path.join(save_home, 'model.pt')}")

##################################################################################################
##################################################################################################
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

    # FOR CROSS DATASET EVALUTATION, we do not consider w_aug
    if log.param.cross_eval and (("ihc" in log.param.dataset) or ("dynahate" in log.param.dataset) or ("sbic" in log.param.dataset)):
        _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None) # train_batch_size, w_aug --- useless for test
    else:
        _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=log.param.w_aug,w_double=log.param.w_double,label_list=None) # train_batch_size, w_aug --- useless for test

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
        if log.param.run_name == "subset":
            log.param.label_size = int(log.param.label_list.split("-")[0])
        ## reseeding before every run while tuning

        if log.param.dataset == "ed":
            log.param.label_size = 32
        elif log.param.dataset == "emoint":
            log.param.label_size = 4
        elif log.param.dataset == "goemotions":
            log.param.label_size = 27
        elif log.param.dataset == "isear":
            log.param.label_size = 7
        elif log.param.dataset == "sst-2":
            log.param.label_size = 2
        elif log.param.dataset == "sst-5":
            log.param.label_size = 5
        elif log.param.dataset == "ihc_pure":
            log.param.label_size = 2
        elif log.param.dataset == "ihc_pure_imp":
            log.param.label_size = 2
        elif log.param.dataset == "dynahate":
            log.param.label_size = 2
        elif log.param.dataset == "sbic":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_imp":
            log.param.label_size = 2
        elif log.param.dataset == "ihc_pure_imp_only":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_imp_only":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_pure":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_pure_imp":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_pure_hard":
            log.param.label_size = 2
        elif log.param.dataset == "sbic_pure_hard_imp":
            log.param.label_size = 2
        
        if not log.param.load:
            cl_train(log)
        else:
            assert log.param.load_dir is not None, "to load a model, log.param.load_dir should be given!!"
            assert not log.param.save, "when loading the model, save is not to be performed"
            cl_test(log)


