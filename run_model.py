import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models import BertGenerator, BertDiscriminator
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import evaluate


class RunModel:

    def __init__(self, device):
        self.device = device

    def train_classifier(self, train_input_loader, val_input_loader, classifier_model, n_epochs=5, lr=1e-4):
        classifier_model.to(self.device)
        optimizer = optim.Adam(classifier_model.parameters(), lr=lr)
        # optimizer = optim.SGD(classifier_model.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            total_train_loss = 0
            classifier_model.train()
            for train_batch_x, train_batch_y in tqdm(train_input_loader):
                optimizer.zero_grad()
                train_batch_y = train_batch_y.to(self.device)
                y_pred = classifier_model(train_batch_x.to(self.device))
                loss = loss_fn(y_pred, train_batch_y)
                total_train_loss += float(loss)
                loss.backward()
                optimizer.step()
            avg_train_loss = total_train_loss / len(train_input_loader)
            classifier_model.eval()
            total_eval_accuracy = 0
            # for val_x_batch, val_y_batch in val_input_loader:
            #     val_y_batch = val_y_batch.to(self.device)
            #     y_pred = classifier_model(val_x_batch.to(self.device))
            #     ce = loss_fn(y_pred, val_y_batch)
            #     acc = (torch.argmax(y_pred, 1) == torch.argmax(val_y_batch, 1)).float().mean()
            #     total_eval_accuracy += float(acc)
            predictions, labels = self.predict(val_input_loader, classifier_model)
            acc, f1, precision, recall, label_f1 = evaluate(labels, predictions)
            # avg_val_accuracy = total_eval_accuracy / len(val_input_loader)
            print("loss", avg_train_loss, epoch, "acc: ", acc, "f1: ", f1, "precision", precision, "recall", recall)
        return classifier_model

    def test_classifier(self, test_input_loader, classifier_model):
        total_eval_accuracy = 0
        classifier_model.eval()
        for val_x_batch, val_y_batch in test_input_loader:
            val_y_batch = val_y_batch.to(self.device)
            y_pred = classifier_model(val_x_batch.to(self.device))
            acc = (torch.argmax(y_pred, 1) == torch.argmax(val_y_batch, 1)).float().mean()
            total_eval_accuracy += float(acc)
        avg_val_accuracy = total_eval_accuracy / len(test_input_loader)
        print("test acc: ", avg_val_accuracy)

    def predict(self, test_input_loader, classifier_model):
        classifier_model.eval()
        predictions, true_labels = [], []
        for val_x_batch, val_y_batch in test_input_loader:
            val_y_batch = val_y_batch.to(self.device)
            y_pred = classifier_model(val_x_batch.to(self.device))
            predictions.append(y_pred.detach().cpu())
            true_labels.append(val_y_batch.detach().cpu())
        predictions = torch.cat(predictions, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        predict_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(true_labels, axis=1)
        return predict_labels, true_labels

    def generator_trainer(self, generator, discriminator, source_train, source_val, batch_num, n_epochs=10,
                          glr=1e-7, dlr=1e-7):
        generator.to(self.device)
        discriminator.to(self.device)
        optimizer_generator = optim.Adam(generator.parameters(), lr=glr)
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=dlr)
        loss_fn = nn.BCELoss()
        for epoch in range(n_epochs):
            source_iter = iter(source_train)
            discriminator_train_loss = 0
            generator_train_loss = 0
            generator.train()
            discriminator.train()
            for batch_index in range(batch_num):
                p = float(batch_index + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
                try:
                    source, _ = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_train)
                    source, _ = next(source_iter)
                noise = torch.rand(source.size())
                source_label = torch.ones(len(source)).float()
                target_label = torch.zeros(len(source)).float()
                source = source.to(self.device)
                noise = noise.to(self.device)
                target = generator(noise)
                target = target.to(self.device)
                optimizer_discriminator.zero_grad()
                pred_true = discriminator(source)
                pred_fake = discriminator(target.detach())
                loss_fake = loss_fn(pred_fake, target_label.to(self.device))
                loss_true = loss_fn(pred_true, source_label.to(self.device))
                loss = loss_fake + loss_true
                discriminator_train_loss += float(loss)
                loss.backward()
                optimizer_discriminator.step()

                optimizer_generator.zero_grad()
                pre_target = discriminator(target)
                loss_fake = loss_fn(pre_target, source_label.to(self.device)) * grl_lambda
                generator_train_loss += float(loss_fake)
                loss_fake.backward()
                optimizer_generator.step()

            avg_dis_loss = discriminator_train_loss / batch_num
            ave_gen_loss = generator_train_loss / batch_num
            discriminator.eval()
            generator.eval()
            total_eval_accuracy1 = 0
            total_eval_accuracy2 = 0
            val_batch_num = int(0.3 * batch_num)
            source_iter = iter(source_val)
            for batch_index in range(val_batch_num):
                try:
                    source, _ = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_val)
                    source, _ = next(source_iter)
                noise = torch.rand(source.size())
                source_label = torch.ones(len(source)).float()
                with torch.no_grad():
                    target = generator(noise.to(self.device))
                    pre_source = discriminator(source.to(self.device)).detach().cpu()
                    pre_target = discriminator(target.to(self.device)).detach().cpu()
                    acc2 = torch.median(pre_target)
                    acc1 = torch.sum(torch.round(pre_source) == source_label) / len(source)
                    total_eval_accuracy1 += float(acc1)
                    total_eval_accuracy2 += float(acc2)
            avg_val_accuracy1 = total_eval_accuracy1 / val_batch_num
            avg_val_accuracy2 = total_eval_accuracy2 / val_batch_num
            # print(epoch, "discriminator loss: ", avg_dis_loss, "generator loss :", ave_gen_loss, "real acc1: ",
            #       avg_val_accuracy1, "fake middle confidence:",
            #       avg_val_accuracy2)
        return generator, discriminator

    def generate_aug_data(self, generator, discriminator, label, size, batch_num, filter=True):
        all_select = []
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        for i in range(batch_num):
            with torch.no_grad():
                noise = torch.rand(size)
                out = generator(noise.to(self.device)).detach().cpu()
                if filter:
                    logits = discriminator(out.to(self.device)).detach().cpu()
                    filter_threshold = torch.median(logits)
                    select_index = (logits > filter_threshold).nonzero(as_tuple=False)[:, 0]
                    select_data = torch.index_select(out, 0, select_index).detach().cpu()
                    all_select.append(select_data.detach().cpu())
                else:
                    all_select.append(out)
        if len(all_select) == 0:
            all_select.append(torch.rand(size))
        all_select = torch.cat(all_select, dim=0)
        labels = torch.stack([label for i in range(all_select.size(0))])
        dataset = TensorDataset(all_select, labels)
        return dataset

    def augment_data(self, bert, aug_limits, original_data, batch_size=20, g_epochs=20, glr=2e-6, dlr=2e-6):
        gan_aug_datasets = []
        print("We have: " + str(len(aug_limits)) + " labels to be augmented")

        for label in tqdm(aug_limits.keys()):
            aug_limit = aug_limits[label]
            sub_ds = original_data[label]
            gtr = BertGenerator(bert)
            dtr = BertDiscriminator(bert)
            source_train = DataLoader(sub_ds, shuffle=True, batch_size=batch_size)
            gtr, dtr = self.generator_trainer(gtr, dtr, source_train, source_train, batch_num=batch_size,
                                              n_epochs=g_epochs, glr=glr, dlr=dlr)
            one_data, one_label = sub_ds[0]
            size = (batch_size, one_data.size(0), one_data.size(1))
            enrich_batch = int(aug_limit / batch_size) * 2
            gan_aug_dataset = self.generate_aug_data(gtr, dtr, one_label, size=size, batch_num=enrich_batch)
            gan_aug_datasets.append(gan_aug_dataset)
        return gan_aug_datasets

    def augment_bunch_data(self, bert, aug_limits, bunch_num, original_data, batch_size=20, g_epochs=20, glr=2e-6,
                           dlr=2e-6):
        gan_aug_datasets = {}
        print("We have: " + str(len(aug_limits)) + " labels to be augmented")
        for i in range(bunch_num):
            gan_aug_datasets[i] = []
        for label in tqdm(aug_limits.keys()):
            sub_ds = original_data[label]
            gtr = BertGenerator(bert)
            dtr = BertDiscriminator(bert)
            source_train = DataLoader(sub_ds, shuffle=True, batch_size=batch_size)
            gtr, dtr = self.generator_trainer(gtr, dtr, source_train, source_train, batch_num=batch_size,
                                              n_epochs=g_epochs, glr=glr, dlr=dlr)
            one_data, one_label = sub_ds[0]
            size = (batch_size, one_data.size(0), one_data.size(1))
            aug_limit_list = aug_limits[label]
            for i in range(bunch_num):
                aug_limit = aug_limit_list[i]
                enrich_batch = int(aug_limit / batch_size) * 2
                gan_aug_dataset = self.generate_aug_data(gtr, dtr, one_label, size=size, batch_num=enrich_batch)
                gan_aug_datasets[i].append(gan_aug_dataset)
        return gan_aug_datasets
