from util import *
from baselines import *

#pol_vocab = set()
#for x in pol_reader():
#    for r in x['responses']:
#        words = nltk.word_tokenize(r)
#        for w in words: pol_vocab.add(w)
#
#fasttext_embeds = load_fasttext_embeddings(pol_vocab)
#fasttext_embed_sum_phi_c = lambda x: embed_sum_phi_c(x, fasttext_embeds)
#kfold_experiment(pol_reader, fit_maxent_classifier, fasttext_embed_sum_phi_c, None, concat_phi_r, 5, None, False)

lower_unigrams_phi = get_unigrams_phi(lower_pol_reader())
kfold_experiment(reader   = lower_pol_reader(),
                 Model    = MaxEntClassifier,
                 phi      = lower_unigrams_phi,
                 folds    = 5,
                 balanced = True)
