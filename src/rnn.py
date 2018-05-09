import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk
from sklearn.metrics import accuracy_score

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
        sigmoids = self.forward(inputs)
        return torch.round(sigmoids)

# Currently hard coded with Adam optimizer and BCE loss
class NNClassifier(SarcasmClassifier):
    def __init__(self, batch_size, max_epochs, balanced_setting, val_proportion,
                 Module, module_args):
        self.model = Module(**module_args)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.balanced_setting = balanced_setting
        self.val_proportion = val_proportion
        self.train_proportion = 1.0 - val_proportion

    def fit(self, features_sets, label_sets):
        n = len(features_sets)
        assert n == len(label_sets)

        n_train_sets = int(n * self.train_proportion)

        X_train_sets = features_sets[:n_train_sets]
        Y_train_sets = label_sets[:n_train_sets]

        X_val_sets = features_sets[n_train_sets :]
        Y_val_sets = label_sets[n_train_sets :]
        Y_val_flat = [features for features_set in Y_val_sets for features in features_set]

        # We treat examples individually for training, so flatten the training data
        X_train_flat = [features for features_set in X_train_sets for features in features_set]
        Y_train_flat = [features for features_set in Y_train_sets for features in features_set]

        X_train = torch.tensor(X_train_flat, dtype=torch.long)
        Y_train = torch.tensor(Y_train_flat, dtype=torch.float32).view(-1,1)

        #TODO: Replace with with-logits version?
        criterion = nn.BCELoss()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_params)

        num_train_batches = len(X_train) // self.batch_size

        best_val_score = 0.0
        best_val_epoch = 0

        for epoch in range(self.max_epochs):
            print("Starting to train on epoch {}".format(epoch))
            self.model.train()

            running_loss = 0.0
            for b in range(num_train_batches):
                inputs = X_train[b*self.batch_size : (b+1)*self.batch_size]
                labels = Y_train[b*self.batch_size : (b+1)*self.batch_size]

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if b % 20 == 19:  # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, b + 1, running_loss / 20))
                    running_loss = 0.0

            self.model.eval()
            val_predictions = self.predict(X_val_sets)
            flat_predictions = [p for pred_set in val_predictions for p in pred_set]
            rate_val_correct = accuracy_score(Y_val_flat, flat_predictions)

            if rate_val_correct > best_val_score:
                best_val_score = rate_val_correct
                best_val_epoch = epoch

            print("Val classification accuracy: {} (best {} from iteration {})".format(
                rate_val_correct, best_val_score, best_val_epoch))


    def predict(self, features_sets):
        if self.balanced_setting:
            return self.predict_balanced(features_sets)
        else:
            return [self.model.predict(torch.tensor(x, dtype=torch.long)) for x in features_sets]

    def predict_balanced(self, features_sets):
        predictions = []
        for features_set in features_sets:
            input = torch.tensor(features_set,dtype=torch.long)
            probs = self.model(input).detach().numpy()
            most_likely = np.argmax(probs)
            indicator = np.zeros(len(probs))
            indicator[most_likely] = 1
            predictions.append(indicator)
        return predictions


