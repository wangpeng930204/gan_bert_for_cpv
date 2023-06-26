import torch


class CPVDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_name, text_name, device, bert, tokenizer, layer_index, cls_rep, max_len, label_encoder,
                 onehot_encoder):
        self.df = df
        self.label = label_name
        self.text = text_name
        self.device = device
        self.bert = bert
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.layer = layer_index
        self.cls_rep = cls_rep
        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder

    def __getitem__(self, index):
        text = self.df.iloc[index][self.text]
        label = self.df.iloc[index][self.label]
        labels = self._encode_labels(label)[0, :]
        hidden_states = self._encode_sentences(text)[0, :, :]
        return hidden_states, labels

    def __len__(self):
        return self.df.shape[0]

    def _encode_sentences(self, sentences):
        encoded_sen = self.tokenizer(sentences, max_length=self.max_len, padding="max_length", truncation=True,
                                     return_tensors="pt")
        encoded_sen = encoded_sen.to(self.device)
        with torch.no_grad():
            model_outputs = self.bert(**encoded_sen)
            if self.cls_rep:
                return model_outputs[2][self.layer][:, 0, :].detach().cpu()
            else:
                return model_outputs[2][self.layer].detach().cpu()

    def _encode_labels(self, labels):
        integer_encoded = self.label_encoder.transform([labels])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.transform(integer_encoded)
        label_tensor = torch.Tensor(onehot_encoded.todense()).float()
        return label_tensor
