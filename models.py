import torch.nn as nn
import torch
import copy
from transformers import AutoModel

bert_name = "GroNLP/bert-base-dutch-cased"

bert = AutoModel.from_pretrained(bert_name)


class BertGenerator(nn.Module):
    def __init__(self):
        super(BertGenerator, self).__init__()
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.hidden = nn.Linear(768 * 2, 768)

    def forward(self, source, noise):
        x = torch.cat([source, noise], dim=-1)
        x = self.hidden(x)
        x = self.bert_encoder11(x)[0]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        # self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        # self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(76800, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x = self.bert_encoder11(x)[0]
        # x = self.bert_encoder12(x)[0]
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return torch.flatten(x)


class MultiClassifier(nn.Module):
    def __init__(self, num_class):
        super(MultiClassifier, self).__init__()
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x
