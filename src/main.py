from util import *
from baselines import *
from rnn import *
from rnn_util import *



#fast_nn_experiment()


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

