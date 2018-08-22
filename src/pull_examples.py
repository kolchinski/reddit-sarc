from rnn_util import *
from test_configs import *

#Makes a list of predictions for the balanced case of either politics or full dataset
def pull_example_predictions(corpus):
    assert(corpus in ('politics', 'full'))
    print("Pulling example predictions on dataset ", corpus)

    configs = (B2, B3, B4) if corpus == 'politics' else (C2, C3, C4)

    print("Loading fasttext embeddings")
    #TODO: load full fasttext!
    fasttext_lookup, fasttext_word_to_idx = load_embeddings_by_index(FASTTEXT_FILE, 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device, flush=True)

    # Use the first of the three model configs for pulling the data - the
    # hyperparameters relevant for the data processing don't depend on which
    # model is used
    print("Building dataset")
    dataset = build_and_split_dataset(word_to_idx=fasttext_word_to_idx, **configs[0])
    train_data = dataset['train_data']
    val_data = dataset['val_data']
    holdout_datas = dataset['holdout_datas']

    points_and_probs = []

    for i in range(len(configs)):
        print("Running on config number ", i)
        hp = configs[i]

        module_args = {'pretrained_weights' :  fasttext_lookup,
                       'hidden_dim'         :  hp['hidden_dim'],
                       'dropout'            :  hp['dropout'],
                       'freeze_embeddings'  :  hp['freeze_embeddings'],
                       'num_rnn_layers'     :  hp['num_rnn_layers'],
                       'ancestor_rnn'       :  False,
                       'second_linear_layer':  hp['second_linear_layer'],
                       'attn_size'          :  hp['attention_size'],
                       'rnn_cell'           :  hp['rnn_cell'],
                       'embed_addressee'    :  False}

        other_args = {k:v for k,v in hp.items() if k not in module_args}

        for arg in ('recall_multiplier', 'author_feature_shape', 'subreddit_feature_shape'):
            if arg not in other_args: other_args[arg] = None
        classifier = NNClassifier(device=device, module_args=module_args, **other_args)

        print("Fitting classifier")
        results = classifier.fit(train_data, val_data, holdout_datas)

        holdout_data = list(holdout_datas.values())[0]
        points, probs = classifier.prediction_probs(holdout_data['X'], holdout_data['X_reversed'],
                            holdout_data['lengths'], holdout_data['author_features'], holdout_data['subreddit_features'])
        points_and_probs.append((points, probs))





