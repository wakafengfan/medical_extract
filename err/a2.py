
import collections
import json
from pathlib import Path
import numpy as np
from configuration.config import data_dir

train = [l.strip().split('\t') for l in (Path(data_dir)/'train.txt').open() if len(l.split('\t')) == 2]
test = [l.strip().split('\t') for l in (Path(data_dir)/'test.txt').open() if len(l.split('\t')) == 2]

train_len = [len(s[0]) for s in train]
test_len = [len(s[0]) for s in test]

print(f'Train max:{max(train_len)}, min:{min(train_len)}, mean:{np.mean(train_len)}, median:{np.median(train_len)}')
print(f'Train max:{max(test_len)}, min:{min(test_len)}, mean:{np.mean(test_len)}, median:{np.median(test_len)}')

train_labels = collections.Counter([s[1] for s in train]).most_common()
test_labels = collections.Counter([s[1] for s in test]).most_common()

print(train_labels)
print(test_labels)







# train_1 = json.load((Path(data_dir)/'train_1.json').open())
# dev_1 = json.load((Path(data_dir)/'dev_1.json').open())
#
#
# train_1_t = [d['text'] for d in train_1]
# dev_1_t = [d['text'] for d in dev_1]
#
# s = set(train_1_t).intersection(dev_1_t)

#print(len(s), len(dev_1_t), len(set(dev_1_t)), len(train_1_t), len(set(train_1_t)), sum([dev_1_t.count(i) for i in s]))
#<class 'tuple'>: (50, 5000, 4060, 17546, 13998, 111)
print('Done')



