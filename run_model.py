import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import BertGenerator, Discriminator
from utils import get_train_val_dataloader
from importlib import reload
import models

reload(models)

from models import BertGenerator, Discriminator


class RunModel:

    def __init__(self, device):
        self.device = device

    def train_classifier(self, train_input_loader, val_input_loader, classifier_model, n_epochs=5, lr=1e-4):
        classifier_model.to(self.device)
        optimizer = optim.Adam(classifier_model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            total_train_loss = 0
            classifier_model.train()
            for train_batch_x, train_batch_y in train_input_loader:
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
            for val_x_batch, val_y_batch in val_input_loader:
                val_y_batch = val_y_batch.to(self.device)
                y_pred = classifier_model(val_x_batch.to(self.device))
                ce = loss_fn(y_pred, val_y_batch)
                acc = (torch.argmax(y_pred, 1) == torch.argmax(val_y_batch, 1)).float().mean()
                total_eval_accuracy += float(acc)
            avg_val_accuracy = total_eval_accuracy / len(val_input_loader)
            print(epoch, "train loss: ", avg_train_loss, "val acc: ", avg_val_accuracy)
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

    def multiclass_generator_trainer(self, generator, discriminator, train_source_dl, val_source_dl, n_epochs=5,
                                     glr=1e-4, dlr=1e-5):

        generator.to(self.device)
        discriminator.to(self.device)
        optimizer_generator = optim.Adam(generator.parameters(), lr=glr)
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=dlr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            d_loss = 0
            g_loss = 0
            generator.train()
            discriminator.train()
            for source, source_label, noise, target_label, _ in train_source_dl:
                source = source.to(self.device)
                noise = noise.to(self.device)
                source_label = source_label.to(self.device)
                target_label = target_label.to(self.device)

                target = generator(source, noise)
                target = target.to(self.device)

                # Train discriminator
                optimizer_discriminator.zero_grad()
                pre_fake = discriminator(target)
                pre_true = discriminator(source)
                loss_fake = loss_fn(pre_fake, target_label.to(self.device))
                loss_true = loss_fn(pre_true, source_label.to(self.device))
                loss = loss_fake + loss_true
                g_loss += float(loss)
                loss.backward()
                optimizer_discriminator.step()
                # Train Generator
                optimizer_generator.zero_grad()

                # Loss measures of Generator's ability
                pre_target = discriminator(target)
                loss_fake = loss_fn(pre_target, source_label.to(self.device))
                d_loss += float(loss_fake)
                # Update generator's weights
                loss_fake.backward()
                optimizer_generator.step()
            average_d_loss = d_loss / len(train_source_dl)
            average_g_loss = g_loss / len(train_source_dl)
            discriminator.eval()
            generator.eval()
            total_eval_accuracy1 = 0
            total_eval_accuracy2 = 0
            for source, source_label, noise, target_label, _ in val_source_dl:
                with torch.no_grad():
                    target = generator(noise.to(self.device), source.to(self.device))
                    pre_source = discriminator(source.to(self.device))
                    pre_target = discriminator(target.to(self.device))
                    print(pre_target)
                    acc1 = (torch.argmax(pre_source.to(self.device), 1) == torch.argmax(source_label.to(self.device),
                                                                                        1)).float().mean()
                    acc2 = (torch.argmax(pre_target.to(self.device), 1) == torch.argmax(target_label.to(self.device),
                                                                                        1)).float().mean()
                    total_eval_accuracy1 += float(acc1)
                    total_eval_accuracy2 += float(acc2)
            avg_val_accuracy1 = total_eval_accuracy1 / len(val_source_dl)
            avg_val_accuracy2 = total_eval_accuracy2 / len(val_source_dl)
            print(epoch, "discriminator loss: ", average_d_loss, "generator loss:", average_g_loss, "real acc1: ",
                  avg_val_accuracy1, "fake acc2:",
                  avg_val_accuracy2)
        return generator

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

    def binary_generator_trainer(self, generator, discriminator, source_train, noise_train, source_val, noise_val,
                                 batch_num, n_epochs=10, glr=1e-7, dlr=1e-7):
        generator.to(self.device)
        discriminator.to(self.device)
        optimizer_generator = optim.Adam(generator.parameters(), lr=glr)
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=dlr)
        loss_fn = nn.BCELoss()

        for epoch in range(n_epochs):
            source_iter = iter(source_train)
            noise_iter = iter(noise_train)
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
                try:
                    noise, _ = next(noise_iter)
                except StopIteration:
                    noise_iter = iter(noise_train)
                    noise, _ = next(noise_iter)
                # bound_l = torch.min(source)
                # bound_r = torch.max(source)
                # noise = (bound_r - bound_l) * torch.rand(source.size()) + bound_l
                # noise = torch.FloatTensor(source.size()).uniform_(torch.min(source), torch.max(source))
                source_label = torch.ones(len(source)).float()
                target_label = torch.zeros(len(source)).float()
                source = source.to(self.device)
                noise = noise.to(self.device)

                target = generator(source, noise)
                target = target.to(self.device)
                optimizer_generator.zero_grad()
                pre_target = discriminator(target)
                loss_fake = loss_fn(pre_target, source_label.to(self.device)) * grl_lambda
                generator_train_loss += float(loss_fake)
                loss_fake.backward()
                optimizer_generator.step()

                optimizer_discriminator.zero_grad()
                pred_true = discriminator(source)
                pred_fake = discriminator(target)
                loss_fake = loss_fn(pred_fake.detach(), target_label.to(self.device))
                loss_true = loss_fn(pred_true, source_label.to(self.device))
                loss = loss_fake + loss_true
                discriminator_train_loss += float(loss)
                loss.backward()
                optimizer_discriminator.step()

            avg_dis_loss = discriminator_train_loss / batch_num
            ave_gen_loss = generator_train_loss / batch_num
            discriminator.eval()
            generator.eval()
            total_eval_accuracy1 = 0
            total_eval_accuracy2 = 0
            val_batch_num = int(0.3 * batch_num)
            source_iter = iter(source_val)
            noise_iter = iter(noise_val)
            for batch_index in range(val_batch_num):
                try:
                    source, _ = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_val)
                    source, _ = next(source_iter)

                try:
                    noise, _ = next(noise_iter)
                except StopIteration:
                    noise_iter = iter(noise_val)
                    noise, _ = next(noise_iter)
                # bound_l = torch.min(source)
                # bound_r = torch.max(source)
                # noise = (bound_r - bound_l) * torch.rand(source.size()) + bound_l
                # noise = torch.FloatTensor(source.size()).uniform_(torch.min(source), torch.max(source))
                source_label = torch.ones(len(source)).float()
                with torch.no_grad():
                    target = generator(source.to(self.device), noise.to(self.device))
                    pre_source = discriminator(source.to(self.device))
                    pre_target = discriminator(target.to(self.device))
                    acc1 = torch.sum(torch.round(pre_source.detach().cpu()) == source_label.detach().cpu()) / len(
                        source)
                    acc2 = torch.sum(torch.round(pre_target.detach().cpu()) == source_label.detach().cpu()) / len(
                        source)
                    total_eval_accuracy1 += float(acc1)
                    total_eval_accuracy2 += float(acc2)
            avg_val_accuracy1 = total_eval_accuracy1 / val_batch_num
            avg_val_accuracy2 = total_eval_accuracy2 / val_batch_num

            print(epoch, "discriminator loss: ", avg_dis_loss, "generator loss :", ave_gen_loss, "real acc1: ",
                  avg_val_accuracy1, "fake acc2:",
                  avg_val_accuracy2)
        return generator

    def gan_augmentation(self, source_train, noise_train, source_val, noise_val, glr, dlr, epoch, aug_num):
        generator = BertGenerator()
        discriminator = Discriminator()
        generator = self.binary_generator_trainer(generator, discriminator, source_train, noise_train, source_val,
                                                  noise_val, glr=glr, dlr=dlr, n_epochs=epoch, batch_num=20)
        fake_data = []
        fake_label = []
        source_iter = iter(source_train)
        noise_iter = iter(noise_train)
        for i in range(aug_num):
            try:
                source, label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train)
                source, label = next(source_iter)
            try:
                noise, _ = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_train)
                noise, _ = next(noise_iter)
            out = generator(source.to(self.device), noise.to(self.device))
            fake_data.append(out.detach().cpu())
            fake_label.append(label.detach().cpu())
        data_tensor = torch.cat(fake_data, dim=0)
        label_tensor = torch.cat(fake_label, dim=0)
        augmented_dataset = TensorDataset(data_tensor, label_tensor)
        return augmented_dataset
