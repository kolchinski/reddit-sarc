import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk

from util import *
from rnn import NNClassifier, SarcasmGRU


# Run a super minimal experiment to make sure the net runs
def fast_nn_experiment():
    embed_lookup, word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
    phi = lambda a,r: response_index_phi(a, r, word_to_idx, max_len=60)
    dataset = build_dataset(pol_reader, phi)

    X = dataset['features_sets']
    Y = dataset['label_sets']

    module_args = {'pretrained_weights': embed_lookup,
                   'hidden_dim':         10,
                   'dropout':            0.0,
                   'freeze_embeddings':  True,
                   'num_rnn_layers':     1}

    classifier = NNClassifier(batch_size=128, max_epochs=10, balanced_setting=True, val_proportion=.05,
                              Module=SarcasmGRU, module_args=module_args)

    classifier.fit(X, Y)
    return classifier


def nn_experiment(embed_lookup, word_to_ix, ):
    pass


#This one ignores ancestors - generates seqs from responses only
def response_index_phi(ancestors, responses, word_to_ix, max_len, tokenizer=nltk.word_tokenize):
    n = len(responses)
    seqs = np.zeros([n, max_len], dtype=np.int_)

    for i, r in enumerate(responses):
        words = tokenizer(r)
        seq_len = min(len(words), max_len)
        seqs[i, : seq_len] = [word_to_ix[w] if w in word_to_ix else 0 for w in words[:seq_len]]

    #return torch.from_numpy(seqs)
    return seqs


# num_to_read means don't bother reading past the first xx lines of the embeddings file
# Vocab means only read embeddings for the set of words in vocab
def load_embeddings_by_index(embeddings_file, num_to_read=None, vocab=None):
    if num_to_read is not None and num_to_read < 1:
        raise ValueError("Must read at least one embedding to get dimensionality!")

    lookup = [['UNK_placeholder']]
    word_to_idx = {}

    with open(embeddings_file) as f:
        for i, l in enumerate(f):
            if num_to_read is not None and i >= num_to_read: break
            idx = i+1 # account for UNK token at beginning
            fields = l.strip().split()
            word = fields[0]
            if vocab and word not in vocab: continue
            vec = np.array(fields[1:], dtype=np.float32)
            lookup.append(vec)
            assert len(lookup) == idx + 1
            word_to_idx[word] = idx

        # Fill in the UNK token now that we know the embedding length
        lookup[0] = np.zeros(len(lookup[1]))
    lookup = np.asarray(lookup, np.float32)
    torch_lookup = torch.from_numpy(lookup)

    return torch_lookup, word_to_idx
