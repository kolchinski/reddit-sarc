import sys

from rnn_util import *


print("Loading glove 50 embeddings", flush=True)
glove_50_lookup, glove_50_word_to_idx = load_embeddings_by_index(GLOVE_FILES[50])
def glove_50_fn(): return (glove_50_lookup, glove_50_word_to_idx)

print("Loading fasttext embeddings", flush=True)
fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE)
def fasttext_fn(): return (fasttext_lookup, fasttext_word_to_idx)

print("Embedding load complete!", flush=True)

fixed_params = {
                'data_reader'  : full_reader,
                'dataset_splitter' : split_dataset_random_plus_politics,
                'lookup_phi'   : response_index_phi,
                'Module'       : SarcasmRNN,
                'batch_size' : 512,
                'max_epochs' : 100,
                'balanced_setting' : True,
                'epochs_to_persist' : 6,
                'verbose' : True,
                'progress_bar' : False,
                'author_phi_creator' : author_index_phi_creator,
                'subreddit_phi_creator' : subreddit_index_phi_creator,
                'early_stopping' : True,
                'output_graphs' : True,
                }


params_to_try = { 'embed_fn'     : [glove_50_fn, fasttext_fn],
                  'max_len' :   [60,100],
                  'freeze_embeddings' : [True,False],
                  'hidden_dim' : [10, 20, 40, 80, 160],
                  'l2_lambda' : [1e-1, 1e-2, 1e-3, 1e-4],
                  'dropout' : [0.1, 0.3, 0.5],
                  'num_rnn_layers' : [2, 3],
                  'lr' : [1e-1, 1e-2, 1e-3, 1e-4],
                  'second_linear_layer': [False, True],
                  'rnn_cell': ['LSTM', 'GRU'],
                  'author_feature_shape_placeholder' : [(None, 1),(None, 10),(None, 20)],
                  'subreddit_embed_dim' : [2, 5, 10],
                  }

crossval_nn_parameters(fixed_params, params_to_try, 30000, '')

sys.exit()



embed_lookup, word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
glove_50_1000_fn = lambda: (embed_lookup, word_to_idx)

model = nn_experiment(embed_fn=glove_50_1000_fn,
                      lookup_phi=response_index_phi,
                      data_reader=pol_reader,
                      balanced_setting=True,
                      #dataset_splitter=split_dataset_random_05,
                      dataset_splitter=split_dataset_random_plus_politics,
                      #recall_multiplier=40,
                      #max_pts=1000,
                      max_len=60,
                      #subreddit_phi_creator=subreddit_index_phi_creator,
                      #subreddit_embed_dim=10,
                      #author_phi_creator=author_comment_counts_phi_creator,
                      #author_feature_shape_placeholder=(2,),
                      #author_phi_creator=author_index_phi_creator,
                      #author_feature_shape_placeholder=(None, 10),
                      Module=SarcasmRNN,
                      rnn_cell='LSTM',
                      hidden_dim=10,
                      dropout=0.1,
                      l2_lambda=1e-4,
                      lr=1e-3,
                      freeze_embeddings=True,
                      num_rnn_layers=2,
                      second_linear_layer=True,
                      batch_size=512,
                      max_epochs=100,
                      epochs_to_persist=3,
                      verbose=True,
                      progress_bar=True)








