import sys

from rnn_util import *



embed_lookup, word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
glove_50_1000_fn = lambda: (embed_lookup, word_to_idx)

model = nn_experiment(embed_fn=glove_50_1000_fn,
                      lookup_phi=response_index_phi,
                      data_reader=pol_reader,
                      dataset_splitter=split_dataset_random_05,
                      balanced_setting=True,
                      #recall_multiplier=None,
                      #max_pts=10000,
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



sys.exit()


'''
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
                  'lr' : [1e-1]}

crossval_nn_parameters(fixed_params, params_to_try, 5, 'first_crossval.txt')


sys.exit()
'''

print("Loading glove 50 embeddings", flush=True)
glove_50_lookup, glove_50_word_to_idx = load_embeddings_by_index(GLOVE_FILES[50])
def glove_50_fn(): return (glove_50_lookup, glove_50_word_to_idx)

print("Loading glove 100 embeddings", flush=True)
glove_100_lookup, glove_100_word_to_idx = load_embeddings_by_index(GLOVE_FILES[100])
def glove_100_fn(): return (glove_100_lookup, glove_100_word_to_idx)

print("Loading glove 200 embeddings", flush=True)
glove_200_lookup, glove_200_word_to_idx = load_embeddings_by_index(GLOVE_FILES[200])
def glove_200_fn(): return (glove_200_lookup, glove_200_word_to_idx)

print("Loading fasttext embeddings", flush=True)
fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE)
def fasttext_fn(): return (fasttext_lookup, fasttext_word_to_idx)

print("Loading Amazon Glove embeddings", flush=True)
amazon_lookup, amazon_word_to_idx = load_embeddings_by_index(GLOVE_AMAZON_FILE)
def amazon_fn(): return (amazon_lookup, amazon_word_to_idx)

print("Embedding load complete!", flush=True)

fixed_params = {
                'data_reader'  : full_reader,
                'Module'       : SarcasmGRU,
                'batch_size' : 512,
                'max_epochs' : 100,
                'balanced_setting' : True,
                'val_proportion' : 0.01,
                'epochs_to_persist' : 3,
                'verbose' : True,
                'progress_bar' : False}


params_to_try = { 'embed_fn'     : [glove_50_fn, glove_100_fn, glove_200_fn, fasttext_fn, amazon_fn],
                  'lookup_phi'   : [response_index_phi, response_with_ancestors_index_phi],
                  'max_len' :   [60,100,150],
                  'freeze_embeddings' : [True,False],
                  'hidden_dim' : [10, 20, 40, 80, 160],
                  'l2_lambda' : [1e-1, 1e-2, 1e-3, 1e-4],
                  'dropout' : [0.1, 0.3, 0.5],
                  'num_rnn_layers' : [1, 2],
                  'lr' : [1e-1, 1e-2, 1e-3, 1e-4],
                  'second_linear_layer': [False, True]}

crossval_nn_parameters(fixed_params, params_to_try, 30000, '')




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
