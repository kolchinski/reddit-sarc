import random
from collections import OrderedDict, defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import nltk

from util import *
from rnn import NNClassifier, SarcasmGRU


def flatten(list_of_lists):
    return [x for l in list_of_lists for x in l]

# Run a super minimal experiment to make sure the net runs
def fast_nn_experiment():

    embed_lookup, word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
    glove_50_1000_fn = lambda: (embed_lookup, word_to_idx)

    model = nn_experiment(glove_50_1000_fn,
                          pol_reader, response_index_phi,
                          max_len=60,
                          author_phi_creator=None,
                          author_feature_shape_placeholder=None,
                          Module=SarcasmGRU,
                          hidden_dim=10,
                          dropout=0.1,
                          l2_lambda=1e-4,
                          lr=1e-3,
                          freeze_embeddings=True,
                          num_rnn_layers=2,
                          second_linear_layer=True,
                          batch_size=128,
                          max_epochs=10,
                          balanced_setting=True,
                          val_proportion=0.05,
                          epochs_to_persist=3,
                          verbose=True,
                          progress_bar=False)


    return model


#TODO: should probably make train and val splits in nn_experiment instead of in rnn.py, fix that...
def author_comment_counts_phi_creator(data_reader, val_proportion):
    num_sarcastic = defaultdict(int)
    num_non_sarcastic = defaultdict(int)

    data = []
    for x in data_reader():
        data.append(x)
    for x in data[:int(len(data)*(1.-val_proportion))]:
        comment_labels = x['labels']
        comment_authors = x['response_authors']
        for a, l in zip(comment_authors, comment_labels):
            if l == 1: num_sarcastic[a] += 1
            else: num_non_sarcastic[a] += 1

    authors = set(num_sarcastic.keys()) | set(num_non_sarcastic.keys())

    return len(authors), lambda author: [num_sarcastic[author], num_non_sarcastic[author]]
    #return {'num_sarcastic_by_author'     : num_sarcastic,
    #        'num_non_sarcastic_by_author' : num_non_sarcastic}

#TODO: should probably make train and val splits in nn_experiment instead of in rnn.py, fix that...
def author_index_phi_creator(data_reader, val_proportion):
    authors = OrderedDict()
    i = 1

    data = []
    for x in data_reader():
        data.append(x)
    for x in data[:int(len(data)*(1.-val_proportion))]:
        comment_authors = x['response_authors']
        for a in comment_authors:
            if a not in authors:
                authors[a] = i
                i += 1


    return i, lambda author: authors[author] if author in authors else 0


def nn_experiment(embed_fn, data_reader, lookup_phi, max_len,
                  author_phi_creator, author_feature_shape_placeholder,
                  Module, hidden_dim, dropout, l2_lambda, lr,
                  freeze_embeddings, num_rnn_layers,
                  second_linear_layer,
                  batch_size, max_epochs, balanced_setting, val_proportion,
                  epochs_to_persist, verbose, progress_bar):

    embed_lookup, word_to_idx = embed_fn()

    if author_phi_creator is not None:
        num_authors, author_phi = author_phi_creator(data_reader, val_proportion)
        if len(author_feature_shape_placeholder) == 2:
            author_feature_shape = (num_authors, author_feature_shape_placeholder[1])
            author_feature_type = torch.long
        elif len(author_feature_shape_placeholder) == 1:
            author_feature_shape = author_feature_shape_placeholder
            author_feature_type = torch.float
        else:
            raise ValueError()
    else: num_authors, author_phi, author_feature_shape = None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device, flush=True)
    embed_lookup = embed_lookup.to(device)

    phi = lambda a,r: lookup_phi(a, r, word_to_idx, max_len=max_len)
    dataset = build_dataset(data_reader, phi, author_phi)
    X = torch.tensor(flatten(dataset['features_sets']), dtype=torch.long).to(device)
    Y = torch.tensor(flatten(dataset['label_sets']), dtype=torch.float).to(device)
    lengths = torch.tensor(flatten(dataset['length_sets']), dtype=torch.long).to(device)
    if author_phi_creator is not None:
        author_features = torch.tensor(flatten(dataset['author_feature_sets']),
                                       dtype=author_feature_type).to(device)
    else: author_features = None

    module_args = {'pretrained_weights':   embed_lookup,
                   'hidden_dim':           hidden_dim,
                   'dropout':              dropout,
                   'freeze_embeddings':    freeze_embeddings,
                   'num_rnn_layers':       num_rnn_layers,
                   'second_linear_layer':  second_linear_layer,}

    classifier = NNClassifier(batch_size=batch_size, max_epochs=max_epochs,
                              epochs_to_persist=epochs_to_persist,verbose=verbose,
                              progress_bar=progress_bar,
                              balanced_setting=balanced_setting,
                              val_proportion=val_proportion,
                              l2_lambda=l2_lambda, lr=lr,
                              author_feature_shape=author_feature_shape,
                              device=device,
                              Module=Module, module_args=module_args)

    best_results = classifier.fit(X, Y, lengths, author_features)
    return best_results #dict of best_val_score and best_val_epoch

#Fixed params should be a dict of key:value pairs
#Params to try should be a dict from keys to lists of possible values
def crossval_nn_parameters(fixed_params, params_to_try, iterations, log_file):
    i = 0
    results = {}
    consecutive_duplicates = 0
    while True:
        cur_params = OrderedDict(fixed_params)
        for k, l in params_to_try.items():
            cur_params[k] = random.choice(l)
        cur_str = '\n'.join(["{}: {}".format(str(k), str(v)) for k,v in cur_params.items()])
        if cur_str in results:
            consecutive_duplicates += 1
        else:
            consecutive_duplicates = 0
            print("Evaluating parameters: \n", cur_str, flush=True)
            cur_results = nn_experiment(**cur_params)
            results[cur_str] =  cur_results
            print("Parameters evaluated: \n{}\n\n".format(cur_results), flush=True)
            i += 1
        if i >= iterations or consecutive_duplicates >= 100 or i%50 == 0:
            best_results = sorted(results.items(), key=lambda pair: pair[1]['best_val_score'], reverse=True)
            print("Best results so far: ", flush=True)
            for k,v in best_results[:20]:
                print(k, flush=True)
                print(v, flush=True)
                print('\n\n', flush=True)
        if i >= iterations or consecutive_duplicates >= 100:
            break


#This one ignores ancestors - generates seqs from responses only
def response_index_phi(ancestors, responses, word_to_ix, max_len, tokenizer=nltk.word_tokenize):
    n = len(responses)
    seqs = np.zeros([n, max_len], dtype=np.int_)
    lengths = []

    for i, r in enumerate(responses):
        words = tokenizer(r)
        seq_len = min(len(words), max_len)
        seqs[i, : seq_len] = [word_to_ix[w] if w in word_to_ix else 0 for w in words[:seq_len]]
        lengths.append(seq_len)

    #return torch.from_numpy(seqs)
    return seqs, lengths


# TODO: Add special separators between ancestors and between ancestors and responses
# When max_len cuts off the ancestor+responses combination, cut off the ancestors first, then
# the end of the response - the responses are much more informative than the ancestors
# TODO: Could also try cutting off the beginning of the response and see if that does better
def response_with_ancestors_index_phi(ancestors, responses, word_to_ix, max_len, tokenizer=nltk.word_tokenize):
    n = len(responses)
    seqs = np.zeros([n, max_len], dtype=np.int_)
    lengths = []
    ancestor_words = []

    for i, a in enumerate(ancestors):
        if i != 0: ancestor_words.append('Ancestor')
        ancestor_words += tokenizer(a)
    ancestor_words.append('Separator')

    for i, r in enumerate(responses):
        response_words = tokenizer(r)
        if len(ancestor_words) + len(response_words) <= max_len:
            words = ancestor_words + response_words
        elif len(response_words) <= max_len:
            spare_words = max_len - len(response_words)
            words = ancestor_words[-spare_words:] + response_words
        else: #the response alone is longer than max_len
            words = response_words[:max_len]

        seq_len = min(len(words), max_len)
        seqs[i, : seq_len] = [word_to_ix[w] if w in word_to_ix else 0 for w in words[:seq_len]]
        lengths.append(seq_len)

    return seqs, lengths


# num_to_read means don't bother reading past the first xx lines of the embeddings file
# Vocab means only read embeddings for the set of words in vocab
def load_embeddings_by_index(embeddings_file, num_to_read=None, vocab=None):
    if num_to_read is not None and num_to_read < 1:
        raise ValueError("Must read at least one embedding to get dimensionality!")

    lookup = [['UNK_placeholder']]
    word_to_idx = {}

    with open(embeddings_file) as f:
        if embeddings_file == FASTTEXT_FILE: next(f) # Skip first line
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
