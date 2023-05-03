import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import TensorDataset


class CreateDataset:
    def __init__(self, sentence_column, label_column, label_encoder, onehot_encoder,
                 bert_output_layer, running_device, bert_name="GroNLP/bert-base-dutch-cased"
                 ):
        self.sentence_column = sentence_column
        self.label = label_column
        self.bert_layer = bert_output_layer
        self.device = running_device

        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(bert_name, config=self.config).to(running_device)

        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder

    def create(self, input_df):
        sentences_list = list(input_df.loc[:, self.sentence_column].values)
        labels_list = list(input_df.loc[:, self.label].values)
        labels = self._encode_labels(labels_list)
        hidden_states = self._encode_sentences(sentences_list)
        dataset = TensorDataset(hidden_states, labels)
        return dataset

    def create_label_dataset(self, input_df):
        dataset = {}
        for label in set(input_df[self.label].values):
            sub_df = input_df[input_df[self.label] == label]
            sub_ds = self.create(sub_df)
            dataset[label] = sub_ds
        return dataset

    def _encode_sentences(self, sentences: list, max_length=100):
        encoded_sentences = []
        for sen in tqdm(sentences, desc=str("Getting bert hidden states from layer: " + str(self.bert_layer))):
            encoded_sen = self.tokenizer(sen, max_length=max_length, padding="max_length", truncation=True,
                                         return_tensors="pt")
            encoded_sen = encoded_sen.to(self.device)
            with torch.no_grad():
                model_outputs = self.bert(**encoded_sen)
                encoded_sentences.append(model_outputs[2][self.bert_layer].detach().cpu())
                # encoded_sentences.append(model_outputs[0].detach().cpu())
        return torch.cat(encoded_sentences, dim=0)
        # return torch.stack(encoded_sentences)

    def _encode_labels(self, labels: list):
        integer_encoded = self.label_encoder.transform(labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.transform(integer_encoded)
        label_tensor = torch.Tensor(onehot_encoded.todense()).float()
        return label_tensor
