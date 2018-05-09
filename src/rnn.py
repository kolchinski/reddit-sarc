import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from baselines import SarcasmClassifier


class SarcasmGRU(nn.Module):
    def __init__(self, pretrained_weights,
                 hidden_dim=300, dropout=0.5, freeze_embeddings=True,
                 num_rnn_layers=1):

        super(SarcasmGRU, self).__init__()

        embedding_dim = pretrained_weights.shape[1]
        self.embeddings = nn.Embedding.from_pretrained(pretrained_weights, freeze=freeze_embeddings)

        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=num_rnn_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_dim*2, 1)

        #self.linear1 = nn.Linear(hidden_dim*2, hidden_dim)
        #self.relu = nn.ReLU()
        #self.linear2 = nn.Linear(hidden_dim, 1)


    def forward(self, inputs, **kwargs):
        x = self.embeddings(inputs)
        #TODO: provide an initial hidden state?
        x, h = self.gru(x)
        x = self.dropout(x[:, -1, :].squeeze())  # just get the last hidden state
        x = self.linear(x)

        #x = self.linear1(x)
        #x = self.relu(x)
        #x = self.linear2(x)

        x = F.sigmoid(x)  # sigmoid output for binary classification
        return x

    def predict(self, inputs):
        sigmoids = self(inputs)
        return torch.round(sigmoids)

# Currently hard coded with Adam optimizer and BCE loss
class NNClassifier(SarcasmClassifier):
    def __init__(self, batch_size, max_epochs, balanced_setting, val_proportion,
                 device, Module, module_args):
        self.model = Module(**module_args).to(device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.balanced_setting = balanced_setting
        self.val_proportion = val_proportion
        self.train_proportion = 1.0 - val_proportion

    # X and Y should be n x max_len and n x 1 tensors respectively
    def fit(self, X, Y):
        n = len(X)
        assert n == len(Y)
        n_train = int(self.train_proportion * n)

        # If we have pairs of points, one of each of which is sarcastic, keep train and val balanced
        if self.balanced_setting:
            assert n%2 == 0
            if n_train % 2 != 0: n_train += 1

        X_train, X_val = X[:n_train], X[n_train:]
        Y_train, Y_val = Y[:n_train].view(-1,1), Y[n_train:].view(-1,1)

        criterion = nn.BCELoss() # TODO: Replace with with-logits version?
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_params)

        num_train_batches = n_train // self.batch_size

        best_val_score = 0.0
        best_val_epoch = 0

        for epoch in range(self.max_epochs):
            print("Starting to train on epoch {}".format(epoch))
            self.model.train()

            running_loss = 0.0
            for b in tqdm(range(num_train_batches)):
                inputs = X_train[b*self.batch_size : (b+1)*self.batch_size]
                labels = Y_train[b*self.batch_size : (b+1)*self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_predictions = self.predict(X_val)
            rate_val_correct = accuracy_score(Y_val, val_predictions)
            if rate_val_correct > best_val_score:
                best_val_score = rate_val_correct
                best_val_epoch = epoch

            print("\nAvg Loss: {}. \nVal classification accuracy: {} \n(Best {} from iteration {})\n\n".format(
                running_loss/num_train_batches, rate_val_correct, best_val_score, best_val_epoch))


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if self.balanced_setting:
                return self.predict_balanced(X)
            else:
                return self.model.predict(X)

    def predict_balanced(self, X):
        probs = self.model(X)
        assert len(probs) % 2 == 0
        n = len(probs) // 2
        predictions = torch.zeros(2*n)
        for i in range(n):
            if probs[2*i] > probs[2*i + 1]: predictions[2*i] = 1
            else: predictions[2*i + 1] = 1
        return predictions


