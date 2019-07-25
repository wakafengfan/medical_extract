import json
from pathlib import Path

from configuration.config import data_dir
train = [(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'train.json').open())]
train_copy = train.copy()

test_0724 = json.load((Path(data_dir)/'test_0724.json').open())
test_0724_text = [''.join(l[0]) for l in test_0724]

for l in train:
    train_text = ''.join(l[0])

    for t in test_0724_text:
        if train_text in t or t in train_text:
            train_copy.remove(l)
            break
print(f'train: {len(train)}, train_copy: {len(train_copy)}')
print(f'train_set: {len(set(train))}, train_copy set: {len(set(train_copy))}')
print('Done')
json.dump(train_copy, (Path(data_dir)/'train_filter_test.json').open('w'), ensure_ascii=False, indent=4)

#train: 17546, train_copy: 17085
#train_set: 14049, train_copy set: 13605


