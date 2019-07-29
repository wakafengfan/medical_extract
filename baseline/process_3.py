import json
from pathlib import Path

from configuration.config import data_dir
train = [(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'train.json').open())]
train_copy = train.copy()

test_0724 = json.load((Path(data_dir)/'test_0724.json').open())
test_0724_texts = [''.join(l[0]) for l in test_0724]

for l in train:
    train_text = ''.join(l[0])

    for t in test_0724_texts:
        if (train_text in t or t in train_text) and len(t) > 10:
            train_copy.remove(l)
            break
print(f'train: {len(train)}, train_copy: {len(train_copy)}')
print(f'train_set: {len(set(train))}, train_copy set: {len(set(train_copy))}')
print('Done')
# json.dump(list(set(train_copy)), (Path(data_dir)/'train_filter_test.json').open('w'), ensure_ascii=False, indent=4)

#train: 17546, train_copy: 17085
#train_set: 14049, train_copy set: 13605

dev = [(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'dev.json').open())]
dev_copy = dev.copy()
train_texts = [''.join(l[0]) for l in train]
for l in dev:
    dev_text = ''.join(l[0])
    for t in test_0724_texts:
        if (dev_text in t or t in dev_text) and len(t)>10:
            dev_copy.remove(l)
            break
    for t in train_texts:
        if (dev_text in t or t in dev_text) and len(t) > 10 and t in dev_copy:
            dev_copy.remove(l)
            break
print(f'dev: {len(dev)}, dev_copy: {len(dev_copy)}')
print(f'dev_set: {len(set(dev))}, dev_copy set: {len(set(dev_copy))}')
print('Done')
# json.dump(list(set(train_copy+dev_copy)), (Path(data_dir)/'train_dev_filter_test.json').open('w'), ensure_ascii=False, indent=4)


train_0724 = [(tuple(l[0]), tuple(l[1])) for l in json.load((Path(data_dir)/'train_0724.json').open())]
train_0724_copy = train_0724.copy()
train_dev_texts = [''.join(l[0]) for l in list(set(train_copy+dev_copy))]
for l in train_0724:
    train_0724_text = ''.join(l[0])
    for t in test_0724_texts:
        if (train_0724_text in t or t in train_0724_text) and len(t)>10:
            train_0724_copy.remove(l)
            break
    for t in train_dev_texts:
        if (train_0724_text in t or t in train_0724_text) and len(t)>10 and t in train_0724_copy:
            train_0724_copy.remove(l)
            break
print(f'train_0724: {len(train_0724)}, train_0724_copy: {len(train_0724_copy)}')
print(f'train_0724_set: {len(set(train_0724))}, train_0724_copy set: {len(set(train_0724_copy))}')

# json.dump(list(set(train_copy+dev_copy+train_0724_copy)), (Path(data_dir)/'train_dev_train0724_filter_test.json').open('w'), ensure_ascii=False, indent=4)


# train: 17546, train_copy: 17302
# train_set: 14049, train_copy set: 13821
# Done
# dev: 5000, dev_copy: 4935
# dev_set: 4073, dev_copy set: 4014
# Done
# train_0724: 9715, train_0724_copy: 9615
# train_0724_set: 9710, train_0724_copy set: 9613
