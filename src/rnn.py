import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk

from baselines import SarcasmClassifier


VOCAB_SIZE = 1000
MAX_LEN = 60  # This should be the max len of the comments, but double check!
USE_CUDA = True  # Set this to False if you don't want to use CUDA


#0 is unk
def get_embed_weights_and_dict(embed_lookup):
    vocab_size = next(iter(embed_lookup.values())).shape[0]
    embed_values = np.zeros((len(embed_lookup) + 1, vocab_size), dtype=np.float32)
    word_to_ix = {}

    for i, (word, embed) in enumerate(embed_lookup.items()):
        word_to_ix[word] = i + 1
        embed_values[i + 1] = embed

    return torch.from_numpy(embed_values), word_to_ix


#This one ignores ancestors - generates seqs from responses only
def word_index_phi(ancestors, responses, word_to_ix, max_len=MAX_LEN):
    n = len(responses)
    seqs = np.zeros([n, max_len], dtype=np.int_)

    for i, r in enumerate(responses):
        words = nltk.word_tokenize(r)
        seq_len = min(len(words), max_len)
        seqs[i, : seq_len] = [word_to_ix[w] if w in word_to_ix else 0 for w in words[:seq_len]]

    #return torch.from_numpy(seqs)
    return seqs




class SarcasmGRU(nn.Module):
    def __init__(self, pretrained_weights,
                 hidden_dim=300, dropout=0.5):

        super(SarcasmGRU, self).__init__()

        embedding_dim = pretrained_weights.shape[1]
        self.embeddings = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)

        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=1, bidirectional=True, batch_first=True)

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


class GRUClassifier(SarcasmClassifier):
    def __init__(self, pretrained_weights):
        self.model = SarcasmGRU(pretrained_weights)


    def fit(self, response_sets, label_sets):
        # Flatten the incoming data since we treat sibling responses as independent
        X = [response for response_set in response_sets for response in response_set]
        Y = [label for label_set in label_sets for label in label_set]
        n = len(X)
        assert n == len(Y)

        n_train = int(.95*n)

        X_train = torch.tensor(X[:n_train], dtype=torch.long)
        Y_train = torch.tensor(Y[:n_train], dtype=torch.float32).view(-1, 1)

        X_val = torch.tensor(X[n_train:], dtype=torch.long)
        Y_val = torch.tensor(Y[n_train:], dtype=torch.float32).view(-1,1)

        val_len = int(len(response_sets)*.05)
        X_val_paired = response_sets[-val_len:]
        Y_val_paired = label_sets[-val_len:]

        #TODO: Replace with with-logits version?
        criterion = nn.BCELoss()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(trainable_params)

        batch_size = 128
        num_train_batches = n_train // batch_size

        best_val_score = 0.0
        best_val_epoch = 0

        for epoch in range(200):
            print("Starting to train on epoch {}".format(epoch))

            running_loss = 0.0
            for b in range(num_train_batches):
                # get the inputs
                inputs = X_train[b*batch_size : (b+1)*batch_size]
                labels = Y_train[b*batch_size : (b+1)*batch_size]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                #print(outputs)
                #print(labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if b % 20 == 19:  # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, b + 1, running_loss / 20))
                    running_loss = 0.0

            #TODO: This is ugly, there's got to be a better way to do cross val
            #outputs = self.model(X_val) > 0.5
            #rate_val_correct = torch.mean((outputs.long() == Y_val.long()).float())
            #if rate_val_correct > best_val_score:
            #    best_val_score = rate_val_correct
            #    best_val_epoch = epoch
            good_predictions = 0
            for i in range(val_len):
                x1, x2 = X_val_paired[i]
                y1, y2 = Y_val_paired[i]
                x1_pred = self.model(torch.tensor([x1], dtype=torch.long))
                x2_pred = self.model(torch.tensor([x2], dtype=torch.long))
                if (x1_pred > x2_pred) == (torch.tensor(y1 == 1,dtype=torch.uint8)): good_predictions += 1
            rate_val_correct = good_predictions / val_len
            if rate_val_correct > best_val_score:
                best_val_score = rate_val_correct
                best_val_epoch = epoch

            print("Val classification accuracy: {} (best {} from iteration {})".format(
                rate_val_correct, best_val_score, best_val_epoch))



    def predict(self, response_sets, balanced = False):
        if balanced:
            return self.predict_balanced(response_sets)
        else:
            return [self.model.predict(x) for x in response_sets]

    def predict_balanced(self, response_sets):
        predictions = []
        for response_set in response_sets:
            probs = self.model.predict_proba(response_set)
            pos_probs = [p[1] for p in probs]
            most_likely = np.argmax(pos_probs)
            indicator = np.zeros(len(pos_probs))
            indicator[most_likely] = 1
            predictions.append(indicator)

        return predictions


