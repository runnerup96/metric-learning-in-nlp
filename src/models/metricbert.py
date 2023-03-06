import torch
from typing import Dict
from copy import deepcopy


class TransformerBasedEncoder(torch.nn.Module):
    def __init__(self, bert, n_last_layers2train):
        super(TransformerBasedEncoder, self).__init__()
        self.bert = deepcopy(bert)
        self.disable_bert_training()

        if 1 <= n_last_layers2train < len(self.bert.encoder.layer):
            self.modules2train = [
                *self.bert.encoder.layer[-n_last_layers2train:],
                self.bert.pooler,
            ]
        elif n_last_layers2train == len(self.bert.encoder.layer):
            self.modules2train = [
                *self.bert.encoder.layer[-n_last_layers2train:],
                self.bert.pooler,
                self.bert.embeddings,
            ]
        elif n_last_layers2train == 0:
            self.modules2train = [self.bert.pooler]
        elif n_last_layers2train == -1:
            self.modules2train = []
        else:
            raise ValueError("Wrong params amount!")

    def disable_bert_training(self):
        for module in self.bert.parameters():
            module.requires_grad = False

    def enable_bert_layers_training(self):
        for module in self.modules2train:
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        anchors: Dict[str, torch.Tensor],
        positives: Dict[str, torch.Tensor],
        negatives: Dict[str, torch.Tensor],
    ):
        anchor_bert_ouput = self.bert(anchors["token_ids"], anchors["attention_mask"])
        pos_bert_output = self.bert(positives["token_ids"], positives["attention_mask"])
        neg_bert_output = self.bert(negatives["token_ids"], negatives["attention_mask"])

        anchor_cls_repr = anchor_bert_ouput["pooler_output"]
        neg_cls_repr = neg_bert_output["pooler_output"]
        pos_cls_repr = pos_bert_output["pooler_output"]

        return anchor_cls_repr, pos_cls_repr, neg_cls_repr
