import pandas as pd
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
import numpy as np


def get_train_val_dataloader(dataset, train_weight=0.7, batch_size=20):
    train_size = int(train_weight * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )
    validation_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )
    print("train size:", train_size)
    print("val size: ", val_size)
    return train_dataloader, validation_dataloader


def plot_confusion_matrix(conf_mat, target_names, annot=True, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, xticklabels=target_names, yticklabels=target_names, annot=annot, fmt="", **kwargs)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    return fig


def confusion_matrix(true_label, prediction, label_encoder, num_class, value_counts):
    target_names = []
    for i in range(num_class):
        full_label = label_encoder.inverse_transform([i])[0]
        short_label = full_label.split()[0] + "-" + str(value_counts[full_label])
        target_names.append(short_label)
    rel_conf_mat = metrics.confusion_matrix(true_label, prediction, normalize="true")
    rel_conf_mat = np.round(rel_conf_mat * 100)
    abs_conf_mat = metrics.confusion_matrix(true_label, prediction)
    annot = abs_conf_mat
    annot = annot.astype(str)
    annot[abs_conf_mat == 0.0] = ""
    fig = plot_confusion_matrix(rel_conf_mat, target_names, annot=annot)
    return fig


def sample_data(all_df, label_name):
    vc = all_df.afdeling.value_counts()
    vc_5000 = vc[vc.values > 3500]
    sub_df_5000 = all_df[all_df[label_name].isin(vc_5000.index)]
    input_df_1 = sub_df_5000.sample(frac=1).groupby(label_name, sort=False).head(3000)
    vc_100 = vc[vc.values < 500]
    vc_100 = vc_100[vc_100.values > 200]
    sub_df_100 = all_df[all_df[label_name].isin(vc_100.index)]
    input_df_2 = sub_df_100.sample(frac=1).groupby(label_name, sort=False).head(200)
    input_df = pd.concat([input_df_1, input_df_2])
    print(input_df.afdeling.value_counts())
    return input_df


def evaluate(true_label, prediction):
    acc = accuracy_score(true_label, prediction)
    f1 = f1_score(true_label, prediction, average="macro")
    label_f1 = f1_score(true_label, prediction, average=None)
    precision = precision_score(true_label, prediction, average="macro")
    recall = recall_score(true_label, prediction, average="macro")
    print('Total ACC: %.3f' % acc)
    print('F1: %.3f' % f1)
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    return acc, f1, precision, recall, label_f1


def decide_aug_labels(balance, aug_limit, value_counts, percent):
    aug_num = {}
    if balance:
        for label in value_counts.index:
            if value_counts[label] < aug_limit:
                aug_num[label] = aug_limit - value_counts[label]
    else:
        for label in value_counts.index:
            aug_num[label] = int(value_counts[label] * percent)
    return aug_num
