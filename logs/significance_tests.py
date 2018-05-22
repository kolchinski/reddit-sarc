import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

dataset_letters = ('B', 'C', 'D', 'F', 'G')
model_numbers = ('2', '3', '4')

f1s_dict = {}

for a in dataset_letters:
    for b in model_numbers:
        print(a+b)
        f1s = []
        with open(a+b+'_TEST_LOG.txt') as f:
            lines = f.readlines()
            for line in lines:
                if line[:3] == 'F1s':
                    cur_f1s = [float(x) for x in line[line.index('[') + 1 : -2].split(',')]
                    f1s += cur_f1s
        print(len(f1s), np.mean(f1s))
        f1s_dict[a+b] = f1s

#print(f1s_dict)

for dataset_letter in dataset_letters:
    print('Evaluating for column {}'.format(dataset_letter))
    for model_pair in combinations(model_numbers, 2):
        print(model_pair)
        idx1, idx2 = dataset_letter + model_pair[0], dataset_letter + model_pair[1]
        print(wilcoxon(f1s_dict[idx1], f1s_dict[idx2]))
        print('\n')
    print('\n')