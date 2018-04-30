# Author: Alex Kolchinski 2018
# Framework functions inspired by and adapted from https://github.com/cgpotts/cs224u

from collections import Counter
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Define a comment as unigram features
def unigrams_phi_c(comment):
    return Counter(nltk.word_tokenize(comment))


# Simply concatenate vectors from responses - hardcoded for balanced case
def concat_phi_r(response_features_pair):
    assert len(response_features_pair) == 2
    # print(response_features_pair[0].shape, response_features_pair[1].shape)
    cat = np.concatenate((response_features_pair[0], response_features_pair[1]))
    return cat




def fit_maxent_classifier(X, y):
    #print(X.shape, y.shape)
    mod = LogisticRegression(fit_intercept=True)
    mod.fit(X, y)
    return mod

def fit_naive_bayes_classifier(X, y):
    #print(X.shape, y.shape)
    mod = MultinomialNB()
    mod.fit(X, y)
    return mod
