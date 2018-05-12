import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from baselines import SarcasmClassifier


class SarcasmGRU(nn.Module):
    def __init__(self, pretrained_weights, device,
                 hidden_dim=300, dropout=0.5, freeze_embeddings=True,
                 num_rnn_layers=1, second_linear_layer=False):

        super(SarcasmGRU, self).__init__()

        self.device = device

        embedding_dim = pretrained_weights.shape[1]
        self.embeddings = nn.Embedding.from_pretrained(pretrained_weights, freeze=freeze_embeddings)

        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=num_rnn_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)


        # Switch between going straight from GRU state to output or
        # putting in an intermediate relu->hidden layer (halving the size)
        self.second_linear_layer = second_linear_layer
        if self.second_linear_layer:
            self.linear1 = nn.Linear(hidden_dim*2, hidden_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, 1)
        else:
            self.linear = nn.Linear(hidden_dim*2, 1)


    # inputs should be B x max_len LongTensor, lengths should be B-length 1D LongTensor
    def forward(self, inputs, lengths, **kwargs):
        embedded_inputs = self.embeddings(inputs)
        #TODO: provide an initial hidden state?
        gru_states, _ = self.gru(embedded_inputs)

        # Select the final hidden state for each trajectory, taking its length into account
        # Using pack_padded_sequence would be even more efficient but would require
        # sorting all of the sequences - maybe later
        batch_size = gru_states.shape[0]
        hidden_size = gru_states.shape[2]

        idx = torch.ones((batch_size, 1, hidden_size), dtype=torch.long).to(self.device) * \
              (lengths - 1).view(-1, 1, 1)
        final_states = torch.gather(gru_states, 1, idx).squeeze()

        dropped_out = self.dropout(final_states)

        if self.second_linear_layer:
            x = self.linear1(dropped_out)
            x = self.relu(x)
            post_linear = self.linear2(x)
        else:
            post_linear = self.linear(dropped_out)

        probs = F.sigmoid(post_linear)  # sigmoid output for binary classification
        return probs

    def predict(self, inputs, lengths):
        sigmoids = self(inputs, lengths)
        return torch.round(sigmoids)


# epochs_to_persist: how many epochs of non-increasing val score to go for
# Currently hard coded with Adam optimizer and BCE loss
class NNClassifier(SarcasmClassifier):
    def __init__(self, batch_size, max_epochs, epochs_to_persist, verbose,
                 balanced_setting, val_proportion,
                 device, Module, module_args):
        self.model = Module(device=device, **module_args).to(device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.epochs_to_persist = epochs_to_persist
        self.verbose = verbose
        self.balanced_setting = balanced_setting
        self.val_proportion = val_proportion
        self.train_proportion = 1.0 - val_proportion

    # X and (Y and lengths) should be n x max_len and n x 1 tensors respectively
    def fit(self, X, Y, lengths):
        n = len(X)
        assert n == len(Y) == len(lengths)
        n_train = int(self.train_proportion * n)

        # If we have pairs of points, one of each of which is sarcastic, keep train and val balanced
        if self.balanced_setting:
            assert n%2 == 0
            if n_train % 2 != 0: n_train += 1

        X_train, X_val = X[:n_train], X[n_train:]
        Y_train, Y_val = Y[:n_train].view(-1,1), Y[n_train:].view(-1,1)
        lens_train, lens_val = lengths[:n_train], lengths[n_train:]

        criterion = nn.BCELoss() # TODO: Replace with with-logits version?
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_params)

        num_train_batches = n_train // self.batch_size

        best_val_score = 0.0
        best_val_epoch = 0

        for epoch in (range(self.max_epochs) if self.verbose else tqdm(range(self.max_epochs))):
            if self.verbose: print("Starting to train on epoch {}".format(epoch))
            self.model.train()

            running_loss = 0.0
            for b in (tqdm(range(num_train_batches)) if self.verbose else range(num_train_batches)):
                X_batch =    X_train[b*self.batch_size : (b+1)*self.batch_size]
                Y_batch =    Y_train[b*self.batch_size : (b+1)*self.batch_size]
                lens_batch = lens_train[b*self.batch_size : (b+1)*self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch, lens_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_predictions = self.predict(X_val, lens_val)
            rate_val_correct = accuracy_score(Y_val, val_predictions)
            if rate_val_correct > best_val_score:
                best_val_score = rate_val_correct
                best_val_epoch = epoch

            if self.verbose:
                print("\nAvg Loss: {}. \nVal classification accuracy: {} \n(Best {} from epoch {})\n\n".format(
                    running_loss/num_train_batches, rate_val_correct, best_val_score, best_val_epoch))

            if self.epochs_to_persist and epoch - best_val_epoch >= self.epochs_to_persist:
                break

        print("\nTraining complete. Best val score {} from epoch {}\n\n".format(
            best_val_score, best_val_epoch))

    # Note: this is not batch-ified; could make it so if it looks like it's being slow
    def predict(self, X, lengths):
        self.model.eval()
        with torch.no_grad():
            if self.balanced_setting:
                return self.predict_balanced(X, lengths)
            else:
                return self.model.predict(X, lengths)

    def predict_balanced(self, X, lengths):
        probs = self.model(X, lengths)
        assert len(probs) % 2 == 0
        n = len(probs) // 2
        predictions = torch.zeros(2*n)
        for i in range(n):
            if probs[2*i] > probs[2*i + 1]: predictions[2*i] = 1
            else: predictions[2*i + 1] = 1
        return predictions


