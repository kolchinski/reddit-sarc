from rnn_util import *

# Politics subreddit, balanced case, no user embeddings
B2 = {
    "dataset_splitter" : split_dataset_random_05,
    "test_splitter" : split_dataset_val_only_05,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader,
    "test_reader" : pol_test_reader,
    "balanced_setting" : True,
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 10,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Politics subreddit, balanced case, Bayesian priors
B3 = {
    "dataset_splitter" : split_dataset_random_05,
    "test_splitter" : split_dataset_val_only_05,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader,
    "test_reader" : pol_test_reader,
    "balanced_setting" : True,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 10,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Politics subreddit, balanced case,  15d user embeddings
B4 = {
    "dataset_splitter" : split_dataset_random_05,
    "test_splitter" : split_dataset_val_only_05,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader,
    "test_reader" : pol_test_reader,
    "balanced_setting" : True,
    "author_phi_creator" : author_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 15),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 160,
    "dropout" : 0.3,
    "l2_lambda" : .02,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : True,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

B4_UNK_5 = B4.copy()
B4_UNK_5['author_phi_creator'] = author_min5_index_phi_creator


# Full dataset, balanced case, no user embeddings
C2 = {
    "dataset_splitter" : split_dataset_random_plus_politics,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : full_reader,
    "test_reader" : full_test_reader,
    "balanced_setting" : True,
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 80,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Full dataset, balanced case, Bayesian priors
C3 = {
    "dataset_splitter" : split_dataset_random_plus_politics,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : full_reader,
    "test_reader" : full_test_reader,
    "balanced_setting" : True,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 80,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Full dataset, balanced case, 15d embeddings
C4 = {
    "dataset_splitter" : split_dataset_random_plus_politics,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : full_reader,
    "test_reader" : full_test_reader,
    "balanced_setting" : True,
    "author_phi_creator" : author_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 15),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 160,
    "dropout" : 0.3,
    "l2_lambda" : .02,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : True,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

C4_UNK_5 = C4.copy()
C4_UNK_5['author_phi_creator'] = author_min5_index_phi_creator

# Politics subreddit, unbalanced case, no user embeddings
D2 = {
    "dataset_splitter" : split_dataset_random_01,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader_unbalanced,
    "test_reader" : pol_test_reader_unbalanced,
    "balanced_setting" : False,
    "recall_multiplier" : 4.,
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 10,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Politics subreddit, unbalanced case, Bayesian priors
D3 = {
    "dataset_splitter" : split_dataset_random_01,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader_unbalanced,
    "test_reader" : pol_test_reader_unbalanced,
    "balanced_setting" : False,
    "recall_multiplier" : 4.,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 10,
    "dropout" : 0.1,
    "l2_lambda" : 1e-4,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : False,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Politics subreddit, unbalanced case, 15d user embeddings
D4 = {
    "dataset_splitter" : split_dataset_random_01,
    "test_splitter" : split_dataset_val_only_01,
    "lookup_phi" : response_index_phi,
    "data_reader" : pol_reader_unbalanced,
    "test_reader" : pol_test_reader_unbalanced,
    "balanced_setting" : False,
    "recall_multiplier" : 4.,
    "author_phi_creator" : author_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 15),
    "max_len" : 60,
    "Module" : SarcasmRNN,
    "attention_size" : None,
    "rnn_cell" : 'GRU',
    "hidden_dim" : 160,
    "dropout" : 0.3,
    "l2_lambda" : .02,
    "lr" : 1e-3,
    "freeze_embeddings" : True,
    "num_rnn_layers" : 1,
    "second_linear_layer" : True,
    "batch_size" : 256,
    "max_epochs" : 100,
    "epochs_to_persist" : 5,
    "early_stopping" : True,
    "verbose" : False,
    "progress_bar" : True,
    "output_graphs" : False
}

# Full unbalanced dataset; same hyperparameters as politics balanced
E2 = C2.copy()
E2['data_reader'] = full_reader_unbalanced
E2['test_reader'] = full_test_reader_unbalanced

E3 = C3.copy()
E3['data_reader'] = full_reader_unbalanced
E3['test_reader'] = full_test_reader_unbalanced

E4 = C4.copy()
E4['data_reader'] = full_reader_unbalanced
E4['test_reader'] = full_test_reader_unbalanced



# AskReddit configured almost exactly same as politics
def askreddit_reader():
    return sarc_reader(FULL_COMMENTS, FULL_TRAIN_BALANCED, False, 'AskReddit')
def askreddit_test_reader():
    return sarc_reader(FULL_COMMENTS, FULL_TEST_BALANCED, False, 'AskReddit')

def askreddit_reader_unbalanced():
    return sarc_reader(FULL_COMMENTS, FULL_TRAIN_UNBALANCED, False, 'AskReddit')
def askreddit_test_reader_unbalanced():
    return sarc_reader(FULL_COMMENTS, FULL_TEST_UNBALANCED, False, 'AskReddit')

F2 = B2.copy()
F2['data_reader'] = askreddit_reader
F2['test_reader'] = askreddit_test_reader

F3 = B3.copy()
F3['data_reader'] = askreddit_reader
F3['test_reader'] = askreddit_test_reader

F4 = B4.copy()
F4['data_reader'] = askreddit_reader
F4['test_reader'] = askreddit_test_reader


G2 = D2.copy()
G2['data_reader'] = askreddit_reader_unbalanced
G2['test_reader'] = askreddit_test_reader_unbalanced

G3 = D3.copy()
G3['data_reader'] = askreddit_reader_unbalanced
G3['test_reader'] = askreddit_test_reader_unbalanced

G4 = D4.copy()
G4['data_reader'] = askreddit_reader_unbalanced
G4['test_reader'] = askreddit_test_reader_unbalanced
G4['l2_lambda'] = .01 # Adjust regularization



test_configs = {
    'B2' : B2,
    'B3' : B3,
    'B4' : B4,
    'B4_UNK_5' : B4_UNK_5,
    'C2' : C2,
    'C3' : C3,
    'C4' : C4,
    'C4_UNK_5' : C4_UNK_5,
    'D2' : D2,
    'D3' : D3,
    'D4' : D4,
    'E2' : E2,
    'E3' : E3,
    'E4' : E4,
    'F2' : F2,
    'F3' : F3,
    'F4' : F4,
    'G2' : G2,
    'G3' : G3,
    'G4' : G4,
}
