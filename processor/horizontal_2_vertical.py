import json
from pathlib import Path

from configuration.config import data_dir



def h2v(fn):
    text_col = []
    label_col = []
    for d in fn:
        tt = d['text']
        ll = d['mention']

        text_new = [c for c in tt]
        label_new = ['O'] * len(text_new)
        if len(ll) == 0:
            text_col.extend(text_new)
            label_col.extend(label_new)
            continue

        for l in ll:
            m, s, t = l
            if m not in tt or t not in ['disease', 'drug', 'symptom', 'diagnosis']:
                continue
            label_new[s] = 'B_' + t
            label_new[s + 1: s + len(m)] = ['I_' + t] * (len(m) - 1)

        text_col.extend(text_new)
        label_col.extend(label_new)

    text, label = [[]], [[]]
    for t,l in zip(text_col, label_col):
        text[-1].append(t)
        label[-1].append(l)
        if t == '。':
            label.append([])
            text.append([])

    label.pop()
    text.pop()

    label_copy = label.copy()
    text_copy = text.copy()

    for tt, ll in zip(text, label):

        if len(tt) > 150:
            text_copy.remove(tt)
            label_copy.remove(ll)

            intervals = [i * 150 for i in range(len(tt) // 150 + 1)]
            if intervals[-1] == len(tt):
                print('special')
                intervals[-1] -= 1
            interval_new = intervals.copy()
            for i, interval in enumerate(intervals):
                if i == 0:
                    continue
                if ll[interval] != 'O':
                    win_size = ll[:interval+1][::-1].index('O')
                    if win_size > 15:
                        print('==== oversize ===')
                        print(f'win_size:{win_size}')
                    window = ll[interval - win_size: interval]
                    O_i = interval - (win_size - window.index('O'))
                    assert ll[O_i] == 'O'
                    interval_new[i] = O_i
            interval_new.append(10000)
            for idx in range(len(interval_new)):
                if idx + 1 == len(interval_new):
                    continue
                s_idx = interval_new[idx]
                e_idx = interval_new[idx + 1]
                if len(tt[s_idx:e_idx]) > 150:
                    print('tst')

                text_copy.append(tt[s_idx:e_idx])
                label_copy.append(ll[s_idx:e_idx])

    new_data = [(t, l) for t, l in zip(text_copy, label_copy)]

    return new_data


ccks2017 = json.load((Path(data_dir)/'ccks2017.json').open())
ccks2019 = json.load((Path(data_dir)/'ccks2019.json').open())

train_ccks2017 = h2v(ccks2017)
train_ccks2019 = h2v(ccks2019)

json.dump(train_ccks2017, (Path(data_dir)/'train_ccks2017.json').open('w'), ensure_ascii=False, indent=4)
json.dump(train_ccks2019, (Path(data_dir)/'train_ccks2019.json').open('w'), ensure_ascii=False, indent=4)

print('Done')
