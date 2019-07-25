import json
from pathlib import Path

from configuration.config import data_dir

train = [l.strip().split('\t') for l in (Path(data_dir)/'train.txt').open() if len(l.split('\t')) == 2]
test = [l.strip().split('\t') for l in (Path(data_dir)/'test.txt').open() if len(l.split('\t')) == 2]

train_new = []
for l in train:
    tt, ll = l
    text_new = [c for c in tt]
    label_new = ['O'] * len(text_new)
    if '@@'not in ll:
        train_new.append((text_new, label_new))
        continue

    ll = ll.replace('ner_label:', '').split('&&')
    for tag in ll:
        e_n, e = tag.split('@@')
        if e not in tt or e_n not in ['disease','drug','symptom','diagnosis']:
            continue
        offset = tt.index(e)
        label_new[offset] = 'B_' + e_n
        label_new[offset+1: offset+len(e)] = ['I_' + e_n] * (len(e)-1)
        
    train_new.append((text_new, label_new))
        
text_new, label_new = zip(*train_new)
label_copy = list(label_new).copy()
text_copy = list(text_new).copy()

for tt, ll in zip(text_new, label_new):

    if len(tt) > 150:
        text_copy.remove(tt)
        label_copy.remove(ll)

        intervals = [i*150 for i in range(len(tt)//150+1)]
        interval_new = intervals.copy()
        for i, interval in enumerate(intervals):
            if i==0:
                continue
            if ll[interval] != 'O':
                window = ll[interval - 10: interval]
                O_i = interval - (10 - window.index('O'))
                assert ll[O_i] == 'O'
                interval_new[i] = O_i
        interval_new.append(10000)
        for idx in range(len(interval_new)):
            if idx+1==len(interval_new):
                continue
            s_idx = interval_new[idx]
            e_idx = interval_new[idx+1]
            if len(tt[s_idx:e_idx]) > 150:
                print('tst')

            text_copy.append(tt[s_idx:e_idx])
            label_copy.append(ll[s_idx:e_idx])

train_new_new = [(t,l) for t, l in zip(text_copy, label_copy)]

                             

test_new = []
for l in test:
    tt, ll = l
    text_new = [c for c in tt]
    label_new = ['O'] * len(text_new)
    if '@@' not in ll:
        test_new.append((text_new, label_new))
        continue

    ll = ll.replace('ner_label:', '').split('&&')
    for tag in ll:
        e_n, e = tag.split('@@')
        if e not in tt or e_n not in ['disease','drug','symptom','diagnosis']:
            continue
        offset = tt.index(e)
        label_new[offset] = 'B_' + e_n
        label_new[offset + 1: offset + len(e)] = ['I_' + e_n] * (len(e)-1)

    test_new.append((text_new, label_new))

print('Done')
# json.dump(test_new, (Path(data_dir)/'test_0724.json').open('w'), ensure_ascii=False, indent=4)
# json.dump(train_new_new, (Path(data_dir)/'train_0724.json').open('w'), ensure_ascii=False, indent=4)