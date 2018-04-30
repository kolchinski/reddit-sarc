# Author: Alex Kolchinski 2018
# Framework functions inspired by and adapted from https://github.com/cgpotts/cs224u

import os
import sys
import json
import csv
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support


POL_DIR = '../SARC/2.0/pol'
POL_COMMENTS = os.path.join(POL_DIR, 'comments.json')
POL_TRAIN_BALANCED = os.path.join(POL_DIR, 'train-balanced.csv')

FASTTEXT_FILE = '../../static/wiki-news-300d-1M-subword.vec'
GLOVE_FILES = {i : '../../static/glove/glove.6B.{}d.txt'.format(i) for i in (50, 100, 200, 300)}
GLOVE_AMAZON_FILE = '../../static/amazon_glove1600.txt'

# If only want to load embeddings for a set vocab, pass the vocab as a set/dict
def load_embeddings(embeddings_file, vocab=None):
    lookup = {}
    with open(embeddings_file) as f:
        for l in f:
            fields = l.strip().split()
            idx = fields[0]
            if vocab and idx not in vocab: continue
            vec = np.array(fields[1:], dtype=np.float32)
            lookup[idx] = vec
    return lookup

def load_glove_embeddings(size, vocab=None):
    return load_embeddings(GLOVE_FILES[size], vocab)

def load_glove_amazon_embeddings(vocab=None):
    return load_embeddings(GLOVE_AMAZON_FILE, vocab)

def load_fasttext_embeddings(vocab=None):
    return load_embeddings(FASTTEXT_FILE, vocab)


def kfold_experiment(reader, model_fit_fn, phi_c, phi_a, phi_r, folds, vectorizer=None, vectorize=True):
    dataset = build_dataset(reader(), phi_c, phi_a, phi_r, vectorizer, vectorize)
    X = dataset['X']
    Y = dataset['Y']

    f1_scores = []

    kf = KFold(folds)
    i = 0
    for train, test in kf.split(dataset['X'], dataset['Y']):
        sys.stdout.write("\rTraining and testing on fold {}".format(i))
        sys.stdout.flush()
        i += 1

        model = model_fit_fn(X[train], Y[train])
        predictions = model.predict(X[test])

        #TODO: binary averaging is only for the balanced case - update this for unbalanced
        precision, recall, f1_score, support = \
            precision_recall_fscore_support(Y[test], predictions, average='binary')
        f1_scores.append(f1_score)

    print("\nOver {} folds, F1 mean is {}, stdev {}".format(
        folds, np.mean(f1_scores), np.std(f1_scores)))


#Make an iterator over training data. If lower, convert everything to lowercase
#TODO: this doesn't seem like the right place to memoize the set of vocab words
#but what is?
def sarc_reader(comments_file, train_file, lower):
    with open(comments_file, 'r') as f:
        comments = json.load(f)

    with open(train_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            ancestors_idx = row[0].split(' ')
            responses_idx = row[1].split(' ')
            labels = row[2].split(' ')

            # Make everything lowercase if that's how the function was called
            transform = lambda x: x.lower() if lower else x
            ancestors =  [transform(comments[r]['text']) for r in ancestors_idx]
            responses =  [transform(comments[r]['text']) for r in responses_idx]

            yield {'ancestors': ancestors,
                   'responses': responses,
                   'labels'   : labels
                   }

#For convenience, predefine balanced politics readers
def pol_reader():
    return sarc_reader(POL_COMMENTS, POL_TRAIN_BALANCED, False)

def lower_pol_reader():
    return sarc_reader(POL_COMMENTS, POL_TRAIN_BALANCED, True)


# Reader is iterator for training data
# phi_c turns comments into features
# phi_a combines ancestor features into summary
# phi_r combines response features into summary
# Note that this is for the "balanced" framing!
# TODO: Initially ignoring ancestors, include them as another vector later
def build_dataset(reader, phi_c, phi_a, phi_r, vectorizer=None, vectorize=True):
    ancestors = []
    responses = []
    labels = []
    for x in reader:
        ancestors.append(x['ancestors'])
        responses.append(x['responses'])
        labels.append(x['labels'])

    #TODO: this is all not well laid out, rewrite this function

    X = []
    Y = []
    feat_dicts = [[], []]
    N = len(ancestors)
    assert N == len(responses) == len(labels)
    for i in range(N):
        assert len(responses[i]) == 2
        feat_dicts[0].append(phi_c(responses[i][0]))
        feat_dicts[1].append(phi_c(responses[i][1]))

        # We only care about the first of the two labels since in the balanced setting
        # they're either 0 1 or 1 0
        Y.append(int(labels[i][0]))

    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=False)
            feat_matrix = vectorizer.fit_transform(feat_dicts[0] + feat_dicts[1])
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(chain(feat_dicts[0], feat_dicts[1]))

        response_pair_feats = [feat_matrix[:N], feat_matrix[N:]]
    else:
        response_pair_feats = feat_dicts

    X = [phi_r((response_pair_feats[0][i], response_pair_feats[1][i])) for i in range(N)]

    return {'X': np.asarray(X),
            'Y': np.asarray(Y),
            'vectorizer': vectorizer,
            'raw_examples': (ancestors, responses, labels)}


