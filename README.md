# reddit-sarc
Sarcasm detection on Reddit corpus from Khodak et al (http://nlp.cs.princeton.edu/SARC/2.0/)

pull_all_data.sh provides commands to download, extract, and move all data to the locations
expected by the code
pull_small_data.sh also does so, but only for a subset

logs/ contains the logs for all of the test runs reported in the paper, as well as the script
logs/significance_tests.py which was used to generate confidence intervals

src/baselines.py contains code to replicate the baselines from Khodak et al's dataset paper

src/main.py is configured for a run to test whichever "spreadsheet cell index" model/dataset
combination is passed on the command line, e.g. python main.py B2. Commented-out sections in
main.py provide examples for other ways to use the code.

src/rnn.py contains the actual RNN class, as well as the helper class used to train and
evaluate the RNN, generate graphs etc

src/rnn_util.py contains numerous helper functions for data transformation, generating
user representations, etc

src/run_askreddit_tests.sh and run_tests.sh contain examples of how to run the final tests

src/test_configs.py contains hyperparameter configurations for all results reported in the paper

src/util.py contains generic helper functions for data reading and processing