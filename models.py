import torch.nn as nn
import torch
import copy


class BertGenerator(nn.Module):
    def __init__(self, bert):
        super(BertGenerator, self).__init__()
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])

    def forward(self, x):
        x = self.bert_encoder11(x)[0]
        return x


class BertDiscriminator(nn.Module):
    def __init__(self, bert):
        super(BertDiscriminator, self).__init__()
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden2 = nn.Linear(768, 384)
        self.hidden3 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(192, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.bert_encoder11(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.hidden3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return torch.flatten(x)


class FCGenerator(nn.Module):
    def __init__(self):
        super(FCGenerator, self).__init__()
        self.hidden = nn.Linear(100, 200)
        self.output = nn.Linear(200, 100)
        self.act = nn.GELU()

    def forward(self, noise):
        x = self.act(self.hidden(noise))
        x = self.output(x)
        return x


class FCDiscriminator(nn.Module):
    def __init__(self):
        super(FCDiscriminator, self).__init__()
        self.hidden1 = nn.Linear(100, 50)
        self.hidden2 = nn.Linear(50, 25)
        self.hidden3 = nn.Linear(25, 5)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(5, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.hidden3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return torch.flatten(x)


class PretrainClassifier(nn.Module):
    def __init__(self, num_class, bert):
        super(PretrainClassifier, self).__init__()
        self.bert_encoder1 = bert.encoder.layer[0]
        self.bert_encoder2 = bert.encoder.layer[1]
        self.bert_encoder3 = bert.encoder.layer[2]
        self.bert_encoder4 = bert.encoder.layer[3]
        self.bert_encoder5 = bert.encoder.layer[4]
        self.bert_encoder6 = bert.encoder.layer[5]
        self.bert_encoder7 = bert.encoder.layer[6]
        self.bert_encoder8 = bert.encoder.layer[7]
        self.bert_encoder9 = bert.encoder.layer[8]
        self.bert_encoder10 = bert.encoder.layer[9]
        self.bert_encoder11 = bert.encoder.layer[10]
        self.bert_encoder12 = bert.encoder.layer[11]
        self.bert_pooler = bert.pooler
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder1(x)[0]
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class EmbeddingClassifier(nn.Module):
    def __init__(self, num_class, bert):
        super(EmbeddingClassifier, self).__init__()
        self.bert_encoder1 = copy.deepcopy(bert.encoder.layer[0])
        self.bert_encoder2 = copy.deepcopy(bert.encoder.layer[1])
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder1(x)[0]
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier1(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier1, self).__init__()
        self.bert_encoder2 = copy.deepcopy(bert.encoder.layer[1])
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier2(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier2, self).__init__()
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder3(x)[0]
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier3(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier3, self).__init__()
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier4(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier4, self).__init__()
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier5(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier5, self).__init__()
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier6(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier6, self).__init__()
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier7(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier7, self).__init__()
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier8(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier8, self).__init__()
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier9(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier9, self).__init__()
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768*2)
        self.hidden2 = nn.Linear(768*2,  768*2)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(768*2, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier10(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier10, self).__init__()
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


class HiddenClassifier11(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier11, self).__init__()
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class HiddenClassifier12(nn.Module):
    def __init__(self, num_class, bert):
        super(HiddenClassifier12, self).__init__()
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x


class CLSClassifier(nn.Module):
    def __init__(self, num_class):
        super(CLSClassifier, self).__init__()
        self.hidden1 = nn.Linear(768, 384)
        self.hidden2 = nn.Linear(384, 192)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(192, num_class)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.output(x))
        return x
