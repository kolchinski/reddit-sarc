from rnn_util import *


# Still need embed_fn!
pol_balanced_defaults = {
    "dataset_splitter" : split_dataset_random_05,
    "data_reader" : pol_reader,
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
    "verbose" : True,
    "progress_bar" : True,
    "output_graphs" : False
}

pol_unbalanced_defaults = {
    "dataset_splitter" : split_dataset_random_05,
    "data_reader" : pol_reader_unbalanced,
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
    "epochs_to_persist" : 3,
    "verbose" : True,
    "progress_bar" : True,
    "output_graphs" : False
}

full_balanced_defaults = {
    "dataset_splitter" : split_dataset_random_plus_politics,
    "data_reader" : full_reader,
    "balanced_setting" : True,
    "subreddit_phi_creator" : subreddit_index_phi_creator,
    "subreddit_embed_dim" : 10,
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
    "epochs_to_persist" : 3,
    "verbose" : True,
    "progress_bar" : True,
    "output_graphs" : False
}

# No author embed; comment only RNN embed
B2 = {
    "lookup_phi" : response_index_phi,
    "ancestor_rnn" : False,
    "author_phi_creator" : None,
    "author_feature_shape_placeholder" : None,
    "embed_addressee" : False,
}

# Bayesian author embed; comment only RNN embed
B3 = {
    "lookup_phi" : response_index_phi,
    "ancestor_rnn" : False,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "embed_addressee" : False,
}

# Bayesian author and addressee embed; comment only RNN embed
B4 = {
    "lookup_phi" : response_index_phi,
    "ancestor_rnn" : False,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "embed_addressee" : True,
}

# Multidimensional author embed; comment only RNN embed
B5 = {
    "lookup_phi" : response_index_phi,
    "ancestor_rnn" : False,
    "author_phi_creator" : author_addressee_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 20),
    "embed_addressee" : False,
}

# Multidimensional author and addresee embed; comment only RNN embed
B6 = {
    "lookup_phi" : response_index_phi,
    "ancestor_rnn" : False,
    "author_phi_creator" : author_addressee_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 20),
    "embed_addressee" : True,
}




# No author embed; ancestor and comment RNN embed
C2 = {
    "lookup_phi" : response_and_ancestor_index_phi,
    "ancestor_rnn" : True,
    "author_phi_creator" : None,
    "author_feature_shape_placeholder" : None,
    "embed_addressee" : False,
}

# Cayesian author embed; ancestor and comment RNN embed
C3 = {
    "lookup_phi" : response_and_ancestor_index_phi,
    "ancestor_rnn" : True,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "embed_addressee" : False,
}

# Cayesian author and addressee embed; ancestor and comment RNN embed
C4 = {
    "lookup_phi" : response_and_ancestor_index_phi,
    "ancestor_rnn" : True,
    "author_phi_creator" : author_comment_counts_phi_creator,
    "author_feature_shape_placeholder" : (2,),
    "embed_addressee" : True,
}

# Multidimensional author embed; ancestor and comment RNN embed
C5 = {
    "lookup_phi" : response_and_ancestor_index_phi,
    "ancestor_rnn" : True,
    "author_phi_creator" : author_addressee_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 20),
    "embed_addressee" : False,
}

# Multidimensional author and addresee embed; ancestor and comment RNN embed
C6 = {
    "lookup_phi" : response_and_ancestor_index_phi,
    "ancestor_rnn" : True,
    "author_phi_creator" : author_addressee_index_phi_creator,
    "author_feature_shape_placeholder" : (None, 20),
    "embed_addressee" : True,
}
