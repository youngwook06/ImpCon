from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class primary_encoder_v2_no_pooler_for_con(nn.Module):

    def __init__(self,hidden_size,emotion_size,encoder_type="bert-base-uncased"):
        super(primary_encoder_v2_no_pooler_for_con, self).__init__()

        if encoder_type == "bert-base-uncased":
            options_name = "bert-base-uncased"
            self.encoder_supcon = BertModel.from_pretrained(options_name,num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False

        elif encoder_type == "hatebert":
            options_name = "hate_bert"
            self.encoder_supcon = BertModel.from_pretrained(options_name,num_labels=emotion_size)
            self.encoder_supcon.encoder.config.gradient_checkpointing=False

        else:
            raise NotImplementedError

        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size,emotion_size)

    def pooler(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_activation(x)
        return x

    def get_cls_features_ptrnsp(self, text, attn_mask):
        supcon_fea = self.encoder_supcon(text,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
        norm_supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:,0,:], dim=1) # normalized last layer's first token ([CLS])
        pooled_supcon_fea_cls = supcon_fea.pooler_output # [huggingface] Last layer hidden-state of the first token of the sequence (classification token) **further processed by a Linear layer and a Tanh activation function.** The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        return pooled_supcon_fea_cls, norm_supcon_fea_cls

    def forward(self, pooled_supcon_fea_cls):
        supcon_fea_cls_logits = self.label(self.pooler_dropout(pooled_supcon_fea_cls))

        return supcon_fea_cls_logits
