import torch
import itertools

################################################################

def collate_fn_ihc(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}
    d["label"] = item_info["label"]
    # d["post"] = post_batch
    d["post"] = post_batch.cuda()
    # d["post_attn_mask"] = post_attn_mask
    d["post_attn_mask"] = post_attn_mask.cuda()



    return d

################################################################

def collate_fn_w_aug_ihc_imp_con(data): # original + augmented (all original posts/labels come first and then all augmented posts/labels comes (e.g. org_posts_1, org_posts_2, ... , org_posts_last, aug_posts_1, ..., aug_posts_last))

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
        flat = itertools.chain.from_iterable(item_info[key])
        original_posts = []
        augmented_posts = []
        for i, one_post in enumerate(flat):
            if i % 2 == 0:
                original_posts.append(one_post)
            else:
                augmented_posts.append(one_post)
        original_n_augmented_posts = original_posts + augmented_posts

        item_info[key] = original_n_augmented_posts

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}

    d["label"] = item_info["label"]
    d["post"] = post_batch
    d["post_attn_mask"] = post_attn_mask

    return d


#################################################################


####################################################################################
#####################################cross dataset##################################
####################################################################################


def collate_fn_dynahate(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            # end = lengths[i]
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}
    d["label"] = item_info["label"]
    # d["post"] = post_batch
    d["post"] = post_batch.cuda()
    # d["post_attn_mask"] = post_attn_mask
    d["post_attn_mask"] = post_attn_mask.cuda()

    return d


def collate_fn_sbic(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = min(lengths[i], N)
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}
    d["label"] = item_info["label"]
    # d["post"] = post_batch
    d["post"] = post_batch.cuda()
    # d["post_attn_mask"] = post_attn_mask
    d["post_attn_mask"] = post_attn_mask.cuda()

    return d


def collate_fn_w_aug_sbic_imp_con(data): # original + augmented (all original posts/labels come first and then all augmented posts/labels comes (e.g. org_posts_1, org_posts_2, ... , org_posts_last, aug_posts_1, ..., aug_posts_last))

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            # end = lengths[i]
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

        flat = itertools.chain.from_iterable(item_info[key])
        original_posts = []
        augmented_posts = []
        for i, one_post in enumerate(flat):
            if i % 2 == 0:
                original_posts.append(one_post)
            else:
                augmented_posts.append(one_post)

        original_n_augmented_posts = original_posts + augmented_posts

        item_info[key] = original_n_augmented_posts

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}

    d["label"] = item_info["label"]
    d["post"] = post_batch
    d["post_attn_mask"] = post_attn_mask

    return d

#####################double#####################
#####################double#####################
#####################double#####################


def collate_fn_w_aug_ihc_imp_con_double(data): # original + augmented_1 (same as single version aug) + augmented_2 (always aug) (all original posts/labels come first and then all augmented posts/labels comes (e.g. org_posts_1, org_posts_2, ... , org_posts_last, aug_posts_1, ..., aug_posts_last))

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            # end = lengths[i]
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

        flat = itertools.chain.from_iterable(item_info[key])
        original_posts = []
        # augmented_posts = []
        aug_sent1_of_post = []
        aug_sent2_of_post = []
        for i, one_post in enumerate(flat):

            if i % 3 == 0:
                original_posts.append(one_post)
            elif i % 3 == 1:
                aug_sent1_of_post.append(one_post)
            else:
                aug_sent2_of_post.append(one_post)

        original_n_augmented_posts = original_posts + aug_sent1_of_post + aug_sent2_of_post

        item_info[key] = original_n_augmented_posts

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}

    d["label"] = item_info["label"]
    d["post"] = post_batch
    d["post_attn_mask"] = post_attn_mask

    return d



###################################################################################
def collate_fn_w_aug_sbic_imp_con_double(data): # original + aug1 + aug2 (all original posts/labels come first and then all augmented posts/labels comes (e.g. org_posts_1, org_posts_2, ... , org_posts_last, aug_posts_1, ..., aug_posts_last))

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: 
            N = 128

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            # end = lengths[i]
            end = min(lengths[i], N)

            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    item_info = {}

    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

        flat = itertools.chain.from_iterable(item_info[key])
        original_posts = []
        # augmented_posts = []
        aug_sent1_of_post = []
        aug_sent2_of_post = []
        for i, one_post in enumerate(flat):
            if i % 3 == 0:
                original_posts.append(one_post)
            elif i % 3 == 1:
                aug_sent1_of_post.append(one_post)
            else:
                aug_sent2_of_post.append(one_post)
        
        original_n_augmented_posts = original_posts + aug_sent1_of_post + aug_sent2_of_post

        item_info[key] = original_n_augmented_posts

    ## input
    post_batch,post_attn_mask, post_lengths = merge(item_info['post'])

    d={}
    d["label"] = item_info["label"]
    d["post"] = post_batch
    d["post_attn_mask"] = post_attn_mask

    return d