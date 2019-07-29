import json
from pathlib import Path
from configuration.config import data_dir

train = json.load((Path(data_dir) / 'test_0724.json').open())

T = []
for l in train:
    tt, ll = l
    mention = []
    B_idx = [i for i, l in enumerate(ll) if 'B' in l]
    for i in B_idx:
        e = tt[i]
        e_n = ll[i]
        j = i + 1
        while j < len(ll):
            if ll[j] == 'O' or 'B' in ll[j]:
                break
            e += tt[j]
            j += 1
        mention.append((e, str(i), e_n.split('_')[-1]))

    T.append({'text': ''.join(tt), 'mention': sorted(mention, key=lambda x: int(x[1]))})

# json.dump(T, (Path(data_dir)/'train_1.json').open('w'), ensure_ascii=False, indent=4)
json.dump(T, (Path(data_dir) / 'test_0724_1.json').open('w'), ensure_ascii=False, indent=4)


def trans_dev():
    dev = json.load((Path(data_dir) / 'dev.json').open())
    D = []
    for l in dev:
        tt, ll = l
        mention = []
        B_idx = [i for i, l in enumerate(ll) if 'B' in l]
        for i in B_idx:
            e = tt[i]
            e_n = ll[i]
            j = i + 1
            while j < len(ll):
                if ll[j] == 'O' or 'B' in ll[j]:
                    break
                e += tt[j]
                j += 1
            mention.append((e, str(i), e_n.split('_')[-1]))

        D.append({'text': ''.join(tt), 'mention': sorted(mention, key=lambda x: int(x[1]))})

    json.dump(D, (Path(data_dir) / 'dev_1.json').open('w'), ensure_ascii=False, indent=4)


def trans_data_v2():
    W = []
    for l in (Path(data_dir) / 'ner_2w_checked.txt').open():
        text, label = l.strip().split('\t')
        label = label.replace('ner_label:', '').split('&&')
        mention = []
        for s in label:
            if len(s.split('@@')) == 2 and s.split('@@')[0] in ['disease', 'drug', 'diagnosis', 'symptom']:
                e_n, e = s.split('@@')
                if e not in text:
                    continue
                offset = text.index(e)
                mention.append((e, str(offset), e_n))
        if len(mention) == 0:
            print('8' * 20)
            continue
        W.append({'text': text, 'mention': sorted(mention, key=lambda x: int(x[1]))})

    json.dump(W, (Path(data_dir) / 'ner_v2.json').open('w'), ensure_ascii=False, indent=4)
    print(f'W size: {len(W)}')  # 19974


print('Done')

# <class 'tuple'>: (1285, 5281, 2071, 11120, 2031, 892)

# len(set([s[0] for s in T]).intersection(set([s[0] for s in D]))),
# len(set([s[0] for s in T])), len(set([s[0] for s in D])),
# len(set(W)),
# len(set([s[0] for s in T]).intersection(set(W))),
# len(set([s[0] for s in D]).intersection(set(W))
