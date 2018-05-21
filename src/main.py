import sys
from copy import deepcopy

from rnn_util import *
from test_configs import *


#print("Loading glove embeddings", flush=True)
#glove_lookup, glove_word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
#glove_50_fn = lambda: (glove_lookup, glove_word_to_idx)

def main():
    arg = sys.argv[1] # should be "B2" or similar
    print("Evaluating on {}".format(arg))

    print("Loading fasttext embeddings", flush=True)
    fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE,1000)
    print("Embed load complete!")

    hp = test_configs[arg].copy()
    hp['dataset_splitter'] = hp['test_splitter']
    dataset = build_and_split_dataset(word_to_idx=fasttext_word_to_idx, **hp)
    hp['dataset_splitter'] = split_dataset_train_only
    hp['data_reader'] = hp['test_reader']
    test_data = build_and_split_dataset(word_to_idx=fasttext_word_to_idx, **hp)
    dataset['holdout_datas'] = {'TEST_SET' : test_data['train_data']}
    #results = experiment_on_dataset(embed_lookup=fasttext_lookup, **hp, **dataset)
    final_f1s, final_accuracies = experiment_n_times(15, fasttext_lookup, **dataset, **hp)

if __name__ == "__main__":
    main()

'''
default_hyperparams =   {
    # Data representation
    'embed_fn'     : glove_50_fn,
    'freeze_embeddings' : True,
    'data_reader'  : pol_reader,
    'max_pts' : None, # Only read this many points from the data reader
    'balanced_setting' : True,
    'recall_multiplier': None, #4ish is (was?) good
    'dataset_splitter' : split_dataset_random_05,
    'lookup_phi'   : response_index_phi,
    'max_len' :   60, # Truncate comments longer than this

    # Architecture
    'Module'       : SarcasmRNN,
    'rnn_cell': 'GRU',
    'num_rnn_layers' : 1,
    'second_linear_layer': False,
    'hidden_dim' :  20,
    'attention_size' : 10,

    # Regularization and learning
    'dropout' :  0.5,
    'l2_lambda' : .02,
    'lr' : .001,
    'batch_size' : 256,

    # Author features
    'author_phi_creator' : None,
    'author_feature_shape_placeholder' : None,
    'embed_addressee': False,

    # Subreddit features
    'subreddit_phi_creator' : None,
    'subreddit_embed_dim' : None,

    # Training config
    'epochs_to_persist' : 5,
    'early_stopping' : True,
    'max_epochs' : 5,

    # Logging and display
    'progress_bar' : True,
    'verbose' : True,
    'output_graphs' : True,
}
'''



#hyperparams = deepcopy(default_hyperparams)
#nn_experiment(**hyperparams)

#dataset = build_and_split_dataset(word_to_idx=glove_word_to_idx, **hyperparams)
#experiment_n_times(3, glove_lookup, **dataset, **hyperparams)

#embed_fns = [fasttext_fn, glove_50_fn]
#data_readers = [pol_reader, full_reader]
#dataset_splitters = [split_dataset_random_01, split_dataset_random_05, split_dataset_random_plus_politics]
#lookup_phis = [response_index_phi, response_with_ancestors_index_phi]
#author_phi_creators = [author_index_phi_creator, author_addressee_index_phi_creator]
#subreddit_phi_creators = [subreddit_index_phi_creator]


'''
embed_lookup, word_to_idx = load_embeddings_by_index(GLOVE_FILES[50], 1000)
glove_50_1000_fn = lambda: (embed_lookup, word_to_idx)

model = nn_experiment(embed_fn=glove_50_1000_fn,
                      data_reader=pol_reader,
                      dataset_splitter=split_dataset_random_05,
                      lookup_phi=response_index_phi,
                      max_len=60,
                      author_phi_creator=author_addressee_index_phi_creator,
                      #author_phi_creator=author_index_phi_creator,
                      embed_addressee=True,
                      author_feature_shape_placeholder=(None, 10),
                      Module=SarcasmRNN,
                      rnn_cell='GRU',
                      hidden_dim=10,
                      attention_size=10,
                      dropout=0.1,
                      l2_lambda=1e-4,
                      lr=1e-3,
                      freeze_embeddings=True,
                      num_rnn_layers=2,
                      second_linear_layer=True,
                      batch_size=256,
                      max_epochs=100,
                      balanced_setting=True,
                      epochs_to_persist=3,
                      verbose=True,
                      progress_bar=True)

sys.exit()



print("Loading fasttext embeddings", flush=True)
fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE)
def fasttext_fn(): return (fasttext_lookup, fasttext_word_to_idx)

model = nn_experiment(embed_fn=fasttext_fn,
                      data_reader=pol_reader,
                      dataset_splitter=split_dataset_random_05,
                      lookup_phi=response_index_phi,
                      max_len=60,
                      author_phi_creator=None,
                      author_feature_shape_placeholder=None,
                      Module=SarcasmRNN,
                      rnn_cell='GRU',
                      hidden_dim=10,
                      dropout=0.1,
                      l2_lambda=1e-4,
                      lr=1e-3,
                      freeze_embeddings=True,
                      num_rnn_layers=2,
                      second_linear_layer=True,
                      batch_size=128,
                      max_epochs=10,
                      balanced_setting=True,
                      epochs_to_persist=3,
                      verbose=True,
                      progress_bar=True)



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

'''
