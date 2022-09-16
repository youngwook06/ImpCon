import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset


from collate_fns_imp_con import collate_fn_ihc, collate_fn_w_aug_ihc_imp_con, collate_fn_dynahate, collate_fn_sbic, collate_fn_w_aug_sbic_imp_con, collate_fn_w_aug_ihc_imp_con_double, collate_fn_w_aug_sbic_imp_con_double


class ihc_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])

class dynahate_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])

class sbic_dataset(Dataset):

    def __init__(self,data,training=True,w_aug=False):

        self.data = data
        self.training = training
        self.w_aug = w_aug


    def __getitem__(self, index):

        item = {}

        if self.training and self.w_aug:
            item["post"] = self.data["tokenized_post"][index]
        else:
            item["post"] = torch.LongTensor(self.data["tokenized_post"][index])

        item["label"] = self.data["label"][index]

        return item

    def __len__(self):
        return len(self.data["label"])


def get_dataloader(train_batch_size,eval_batch_size,dataset,seed=None,w_aug=True,w_double=False,label_list=None):
    if w_aug:
        if w_double:
            with open('./preprocessed_data/'+dataset+'_double_syn_preprocessed_bert.pkl', "rb") as f:
                data = pickle.load(f)
        else:
            with open('./preprocessed_data/'+dataset+'_waug_syn_preprocessed_bert.pkl', "rb") as f:
                data = pickle.load(f)
    else:
        with open('./preprocessed_data/'+dataset+'_preprocessed_bert.pkl', "rb") as f:
            data = pickle.load(f)

    if "ihc" in dataset:
        train_dataset = ihc_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = ihc_dataset(data["valid"],training=False,w_aug=w_aug)
        test_dataset = ihc_dataset(data["test"],training=False,w_aug=w_aug)
    elif "dynahate" in dataset:
        train_dataset = dynahate_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = dynahate_dataset(data["dev"],training=False,w_aug=w_aug)
        test_dataset = dynahate_dataset(data["test"],training=False,w_aug=w_aug)
    elif "sbic" in dataset:
        train_dataset = sbic_dataset(data["train"],training=True,w_aug=w_aug)
        valid_dataset = sbic_dataset(data["dev"],training=False,w_aug=w_aug)
        test_dataset = sbic_dataset(data["test"],training=False,w_aug=w_aug)
    else:
        raise NotImplementedError

    if "ihc" in dataset:
        collate_fn = collate_fn_ihc
        if w_double:
            collate_fn_w_aug = collate_fn_w_aug_ihc_imp_con_double # original1, original2, .... aug1, aug2 
        else:
            collate_fn_w_aug = collate_fn_w_aug_ihc_imp_con # original1, original2, .... aug1, aug2 
    elif "dynahate" in dataset:
        assert not w_aug, "for cross dataset evaluation, we do not consider w_aug"
        collate_fn = collate_fn_dynahate
        collate_fn_w_aug = collate_fn_dynahate # original1, original2, .... aug1, aug2 
    elif "sbic" in dataset:
        # assert not w_aug, "for cross dataset evaluation, we do not consider w_aug"
        # EXCEPT FOR SBIC, WHICH IS USED FOR TRAIN AS WELL
        collate_fn = collate_fn_sbic
        if w_double:
            collate_fn_w_aug = collate_fn_w_aug_sbic_imp_con_double # original1, original2, .... aug1, aug2 
        else:
            collate_fn_w_aug = collate_fn_w_aug_sbic_imp_con # original1, original2, .... aug1, aug2 
    else:
        raise NotImplementedError

    if w_aug:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn_w_aug,num_workers=0)
    else:
        train_iter  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)

    valid_iter  = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0)

    test_iter  = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=0)


    return train_iter,valid_iter,test_iter

