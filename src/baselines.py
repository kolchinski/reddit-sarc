
from collections import Counter
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from util import get_reader_vocab

# Phi function should combine m ancestors and n responses into a list of n featurized representations
# This one simply ignores ancestors and does a unigram representation of responses
def unigrams_phi_vectorized(ancestors, responses, vectorizer):
    response_phi = lambda r: \
        np.squeeze(vectorizer.transform(Counter(nltk.word_tokenize(r))))
    return [response_phi(response) for response in responses]

# Pretrains vectorization
def get_unigrams_phi(reader):
    responses_counts = [Counter(nltk.word_tokenize(r)) for x in reader for r in x['responses']]
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit(responses_counts)
    return lambda a,r: unigrams_phi_vectorized(a, r, vectorizer)


#This one ignores ancestors and embeds responses as vector sums
def embed_sum_phi(ancestors, responses, embeddings):
    featurizations = []
    unk = np.zeros(next(iter(embeddings.values())).shape)
    for r in responses:
        words = nltk.word_tokenize(r)
        featurization = np.sum([embeddings[w] if w in embeddings else unk for w in words], axis=0)
        featurizations.append(featurization)
    return featurizations

def get_embed_sum_phi(embeddings):
    return lambda a, r: embed_sum_phi(a, r, embeddings)

def get_embeddings_and_sum_phi(reader, get_embed_fn):
    vocab = get_reader_vocab(reader)
    embeds = get_embed_fn(vocab)
    return get_embed_sum_phi(embeds)


# By the time examples are passed to classifiers for training or classification,
# they should already have contextual information included in their representations
# such that each x and y are effectively independent of each other
# Nevertheless, to achieve compatibility with the balanced case, we require fitting
# and prediction to take lists of lists - sibling comments go in together
class SarcasmClassifier():
    def __init__(self):
        raise NotImplementedError

    def fit(self, response_sets, label_sets):
        raise NotImplementedError()

    def predict(self, response_sets, balanced = False):
        raise NotImplementedError()

    # Assume only one of each set of comments is sarcastic
    def predict_balanced(self, response_sets):
        raise NotImplementedError()


class MaxEntClassifier(SarcasmClassifier):
    def __init__(self, c=1.0):
        self.model = LogisticRegression(C=c)

    def fit(self, response_sets, label_sets):
        # Flatten the incoming data since we treat sibling responses as independent
        X = [response for response_set in response_sets for response in response_set]
        Y = [label for label_set in label_sets for label in label_set]
        assert len(X) == len(Y)
        self.model.fit(X, Y)

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



