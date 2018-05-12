# Author: Alex Kolchinski 2018
# Framework functions inspired by and adapted from https://github.com/cgpotts/cs224u

import os
import sys
import json
import csv
import nltk
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

POL_DIR = '../SARC/2.0/pol'
POL_COMMENTS = os.path.join(POL_DIR, 'comments.json')
POL_TRAIN_BALANCED = os.path.join(POL_DIR, 'train-balanced.csv')
#POL_TEST_BALANCED = os.path.join(POL_DIR, 'test-balanced.csv')

FULL_DIR = '../SARC/2.0/main'
FULL_COMMENTS = os.path.join(FULL_DIR, 'comments.json')
FULL_TRAIN_BALANCED = os.path.join(FULL_DIR, 'train-balanced.csv')

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


# Reader should be initialized instance
# Model should be the class itself - maybe revise this later if would help generality?
def kfold_experiment(reader, Model, phi, folds, balanced=False):
    dataset = build_dataset(reader, phi)
    response_sets = dataset['features_sets']
    label_sets = dataset['label_sets']
    assert len(response_sets) == len(label_sets)
    N = len(response_sets)

    f1_scores = []

    kf = KFold(folds)
    i = 0
    #for train, test in kf.split(response_sets, label_sets):
    for train, test in kf.split(range(N)):
        sys.stdout.write("\rTraining and testing on fold {}".format(i))
        sys.stdout.flush()
        i += 1

        model = Model()
        model.fit([response_sets[i] for i in train],
                  [label_sets[i] for i in train])
        predictions = model.predict([response_sets[i] for i in test], balanced=balanced)

        flat_predictions = [p for pred_set in predictions for p in pred_set]
        test_labels = [label_sets[i] for i in test]
        flat_labels = [l for label_set in test_labels for l in label_set]

        #TODO: binary averaging is only for the balanced case - update this for unbalanced
        precision, recall, f1_score, support = \
            precision_recall_fscore_support(flat_labels, flat_predictions, average='binary')
        f1_scores.append(f1_score)

    print("\nOver {} folds, F1 mean is {}, stdev {}".format(
        folds, np.mean(f1_scores), np.std(f1_scores)))

def train_and_eval(train_reader, test_reader, Model, phi, balanced=False):
    train_dataset = build_dataset(train_reader, phi)
    train_response_sets = train_dataset['features_sets']
    train_label_sets = train_dataset['label_sets']

    test_dataset = build_dataset(test_reader, phi)
    test_response_sets = test_dataset['features_sets']
    test_label_sets = test_dataset['label_sets']

    model = Model()
    model.fit(train_response_sets, train_label_sets)

    flat_labels = [l for label_set in test_label_sets for l in label_set]
    predictions = model.predict(test_response_sets, balanced=balanced)
    flat_predictions = [p for pred_set in predictions for p in pred_set]
    print(classification_report(flat_labels, flat_predictions, digits=3))


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
            labels = np.array(row[2].split(' '), dtype=np.int32)

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

def full_reader():
    return sarc_reader(FULL_COMMENTS, FULL_TRAIN_BALANCED, False)

#def pol_test_reader():
#    return sarc_reader(POL_COMMENTS, POL_TEST_BALANCED, False)

def lower_pol_reader():
    return sarc_reader(POL_COMMENTS, POL_TRAIN_BALANCED, True)

def get_reader_vocab(reader):
    vocab = set()
    for x in reader():
        for r in x['responses']:
            words = nltk.word_tokenize(r)
            for w in words: vocab.add(w)
    return vocab


# Reader is iterator for training data
# Phi is the transform function
# Note that this is for the "balanced" framing!
def build_dataset(reader, phi):
    features_sets = []
    label_sets = []
    length_sets = []

    for x in reader():
        label_sets.append(x['labels'])
        features_set, lengths = phi(x['ancestors'], x['responses'])
        features_sets.append(features_set)
        length_sets.append(lengths)

    return {'features_sets': features_sets,
            'label_sets': label_sets,
            'length_sets': length_sets
            }


