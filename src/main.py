import sys

from util import *
from baselines import *
from rnn import *
from rnn_util import *

#fast_nn_experiment()
#sys.exit()

glove_50_lookup, glove_50_word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
def glove_50_1000_fn(): return (glove_50_lookup, glove_50_word_to_idx)
fixed_params = {'embed_fn'     : glove_50_1000_fn,
                'data_reader'  : pol_reader,
                'lookup_phi'   : response_index_phi,
                'Module'       : SarcasmGRU,
                'freeze_embeddings' : True,
                'batch_size' : 512,
                'max_epochs' : 100,
                'balanced_setting' : True,
                'val_proportion' : 0.05,
                'epochs_to_persist' : 3,
                'verbose' : True,
                'progress_bar' : False,

                'max_len' : 60,
                'hidden_dim' : 10,
                'dropout' : 0.1,
                #'l2_lambda' : 1e-4,
                #'lr' : 1e-3,
                'num_rnn_layers' : 1,
                'second_linear_layer': False}

params_to_try = { 'l2_lambda' : [1e-3, 1e-4],
                  'lr' : [1e-1, 1e-2, 1e-3, 1e-4]}

crossval_nn_parameters(fixed_params, params_to_try, 5, 'first_crossval.txt')




'''
print("Reading embeddings")
fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE, 1000)
print("Embeddings read complete!")

model = nn_experiment(fasttext_lookup, fasttext_word_to_idx, full_reader, response_index_phi,
                      max_len=60,
                      Module=SarcasmGRU,
                      hidden_dim=200,
                      dropout=0.5,
                      freeze_embeddings=True,
                      num_rnn_layers=1,
                      batch_size=128,
                      max_epochs=30,
                      balanced_setting=True,
                      val_proportion=0.01)
                  
model = nn_experiment(embed_lookup, word_to_idx,
                      pol_reader, response_with_ancestors_index_phi,
                      max_len=120,
                      Module=SarcasmGRU,
                      hidden_dim=200,
                      dropout=0.5,
                      freeze_embeddings=True,
                      num_rnn_layers=1,
                      second_linear_layer=False,
                      batch_size=128,
                      max_epochs=30,
                      balanced_setting=True,
                      val_proportion=0.05)

'''
