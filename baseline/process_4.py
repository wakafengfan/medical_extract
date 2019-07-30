import json
from pathlib import Path

from tqdm import tqdm

from configuration.config import data_dir
from difflib import SequenceMatcher

train = list(set([(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'train.json').open())]))
train_copy = train.copy()

test_0724 = json.load((Path(data_dir)/'test_0729.json').open())
test_0724_texts = [''.join(l[0]) for l in test_0724]

for l in tqdm(train):
    train_text = ''.join(l[0])
    for t in test_0724_texts:
        sub_match = SequenceMatcher(None, t, train_text).find_longest_match(0, len(t), 0, len(train_text))
        if sub_match.size / max(len(t), len(train_text)) > 0.7:
            train_copy.remove(l)
            break
print(f'train: {len(train)}, train_copy: {len(train_copy)}')
print(f'train_set: {len(set(train))}, train_copy set: {len(set(train_copy))}')
print('Done')
# json.dump(list(set(train_copy)), (Path(data_dir)/'train_filter_test.json').open('w'), ensure_ascii=False, indent=4)

#train: 17546, train_copy: 17085
#train_set: 14049, train_copy set: 13605

dev = list(set([(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'dev.json').open())]))
dev_copy = dev.copy()
train_texts = [''.join(l[0]) for l in train]
for l in tqdm(dev):
    dev_text = ''.join(l[0])
    for t in test_0724_texts:
        sub_match = SequenceMatcher(None, t, dev_text).find_longest_match(0,len(t),0,len(dev_text))
        if sub_match.size / max(len(t),len(dev_text)) > 0.7:
            dev_copy.remove(l)
            break
    for t in train_texts:
        sub_match = SequenceMatcher(None, t, dev_text).find_longest_match(0,len(t),0,len(dev_text))
        if sub_match.size / max(len(t),len(dev_text)) > 0.7 and l in dev_copy:
            dev_copy.remove(l)
            break
print(f'dev: {len(dev)}, dev_copy: {len(dev_copy)}')
print(f'dev_set: {len(set(dev))}, dev_copy set: {len(set(dev_copy))}')
print('Done')
# json.dump(list(set(train_copy+dev_copy)), (Path(data_dir)/'train_dev_filter_test.json').open('w'), ensure_ascii=False, indent=4)


train_0724 = list(set([(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'train_0729.json').open())]))
train_0724_copy = train_0724.copy()
train_dev_texts = [''.join(l[0]) for l in list(set(train_copy+dev_copy))]
for l in tqdm(train_0724):
    train_0724_text = ''.join(l[0])
    for t in test_0724_texts:
        sub_match = SequenceMatcher(None, t, train_0724_text).find_longest_match(0, len(t), 0, len(train_0724_text))
        if sub_match.size / max(len(t), len(train_0724_text)) > 0.7:
            train_0724_copy.remove(l)
            break
    for t in train_dev_texts:
        sub_match = SequenceMatcher(None, t, train_0724_text).find_longest_match(0, len(t), 0, len(train_0724_text))
        if sub_match.size / max(len(t), len(train_0724_text)) > 0.7 and t in train_0724_copy:
            train_0724_copy.remove(l)
            break
print(f'train_0724: {len(train_0724)}, train_0724_copy: {len(train_0724_copy)}')
print(f'train_0724_set: {len(set(train_0724))}, train_0724_copy set: {len(set(train_0724_copy))}')

json.dump(list(set(train_copy+dev_copy+train_0724_copy)), (Path(data_dir)/'train_dev_train0729_filter_test.json').open('w'), ensure_ascii=False, indent=4)


# train: 17546, train_copy: 17302
# train_set: 14049, train_copy set: 13821
# Done
# dev: 5000, dev_copy: 4935
# dev_set: 4073, dev_copy set: 4014
# Done
# train_0724: 9715, train_0724_copy: 9615
# train_0724_set: 9710, train_0724_copy set: 9613

# 0729
# train_set: 14049, train_copy set: 13768
# dev: 4073, dev_copy: 3411
# train_0724_set: 9710, train_0724_copy set: 9514
