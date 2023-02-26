from copy import deepcopy
from typing import List, Dict
import torch
import pytorch_lightning as pl


class MetricBERT(torch.nn.Module):
    def __init__(self, bert_model: torch.nn.Module):
        super().__init__()
        self.bert = deepcopy(bert_model)

    def _disable_layers_training(self, layers: List[torch.nn.Module]) -> None:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _enable_layers_training(self, layers: List[torch.nn.Module]) -> None:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def disable_bert_training(self) -> None:
        """Disable gradients for the entire model"""
        self._disable_layers_training(self.bert.encoder.layer)

    def enable_some_bert_layers_training(self, layers2freeze: int) -> None:
        """Disable gradients for the first layers2freeze layers

        Args:
            layers2freeze (int): number of first layers to freeze
        """
        layers4freeze = [
            *[self.bert.encoder.layer[:layers2freeze]] + [self.bert.embeddings]
        ]
        layers2train = [
            *self.bert.encoder.layer[layers2freeze - len(self.bert.encoder.layer) :]
        ]
        self._disable_layers_training(layers4freeze)
        self._enable_layers_training(layers2train)

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
