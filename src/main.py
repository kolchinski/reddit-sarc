from util import *
from baselines import *
from rnn import *




vocab = get_reader_vocab(pol_reader)

#fasttext_lookup = load_fasttext_embeddings(vocab)
#embed_weights, word_to_ix = get_embed_weights_and_dict(fasttext_lookup)

glove_amazon_lookup = load_glove_amazon_embeddings(vocab)
embed_weights, word_to_ix = get_embed_weights_and_dict(glove_amazon_lookup)


phi = lambda a,r: word_index_phi(a, r, word_to_ix)
dataset = build_dataset(pol_reader(), phi)

classifier = GRUClassifier(embed_weights)

X = dataset['response_feature_sets']
Y = dataset['label_sets']
print(len(X), len(Y))
classifier.fit(X, Y)
