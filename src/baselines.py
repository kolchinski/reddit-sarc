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

def bigrams_phi_c(comment):
    words = nltk.word_tokenize(comment)
    return Counter(zip(["<s>"] + words, words + ["</s>"]))



def embed_sum_phi_c(comment, embeddings):
    words = nltk.word_tokenize(comment)
    unk = np.zeros(next(iter(embeddings.values())).shape)
    return np.sum([embeddings[w] if w in embeddings else unk for w in words], axis=0)



# Simply concatenate vectors from responses - hardcoded for balanced case
def concat_phi_r(response_features_pair):
    assert len(response_features_pair) == 2
    # print(response_features_pair[0].shape, response_features_pair[1].shape)
    cat = np.concatenate((response_features_pair[0], response_features_pair[1]))
    return cat





def fit_maxent_classifier(X, y, C=1.0):
    #print(X.shape, y.shape)
    mod = LogisticRegression(fit_intercept=True, C=C)
    mod.fit(X, y)
    return mod

def fit_naive_bayes_classifier(X, y):
    #print(X.shape, y.shape)
    mod = MultinomialNB()
    mod.fit(X, y)
    return mod
