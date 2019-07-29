import json
import random
from pathlib import Path
import numpy as np
from configuration.config import data_dir
from collections import Counter

from configuration.dic import sub_dic

text, label = [[]], [[]]


# p = (Path(data_dir)/'ner_total_99w.txt').open()
p = json.load((Path(data_dir)/'train.json').open())

text_new, label_new = zip(*p)

# for l in p:
#     # c, t = l.strip().split()
#     c, t = l
#     label[-1].append(t)
#     text[-1].append(c)
#     if c == '。':
#         label.append([])
#         text.append([])
#
# label.pop()
# text.pop()
#
# label_new = label.copy()
# text_new = text.copy()
#
# for tt, ll in zip(text, label):
#
#     if len(tt) > 150:
#         text_new.remove(tt)
#         label_new.remove(ll)
#
#         intervals = [i*150 for i in range(len(tt)//150+1)]
#         interval_new = intervals.copy()
#         for i, interval in enumerate(intervals):
#             if i==0:
#                 continue
#             if ll[interval] != 'O':
#                 window = ll[interval - 10: interval]
#                 O_i = interval - (10 - window.index('O'))
#                 assert ll[O_i] == 'O'
#                 interval_new[i] = O_i
#         interval_new.append(10000)
#         for idx in range(len(interval_new)):
#             if idx+1==len(interval_new):
#                 continue
#             s_idx = interval_new[idx]
#             e_idx = interval_new[idx+1]
#             if len(tt[s_idx:e_idx]) > 150:
#                 print('tst')
#
#             text_new.append(tt[s_idx:e_idx])
#             label_new.append(ll[s_idx:e_idx])


# length
length = list(map(len, text_new))
print(f'max:{max(length)}, min:{min(length)}, mean:{np.mean(length)}, median:{np.median(length)}')

# labels
col = [t for ts in label_new for t in ts if "B" in t]
print(f'标签分布: {Counter(col)}')


e_cnt = [len([t for t in ts if "B" in t]) for ts in label_new]
print(f'包含实体个数的分布：{Counter(e_cnt)}')

# label_new = [[sub_dic[l] for l in ls] for ls in label_new]
t_l = [(t,l) for t, l in zip(text_new, label_new)]
dev = random.sample(t_l, k=5000)
train = [k for k in t_l if k not in dev]

dev_t, dev_l = zip(*dev)
train_t, train_l = zip(*train)

dev_col = [t for ts in dev_l for t in ts if "B" in t]
print(Counter(dev_col))


train_col = [t for ts in train_l for t in ts if "B" in t]
print(Counter(train_col))

dev_e_cnt = [len([t for t in ts if "B" in t]) for ts in dev_l]
print(Counter(dev_e_cnt))

train_e_cnt = [len([t for t in ts if "B" in t]) for ts in train_l]
print(Counter(train_e_cnt))

# json.dump(dev, (Path(data_dir)/'dev.json').open('w'), ensure_ascii=False)
# json.dump(train, (Path(data_dir)/'train.json').open('w'), ensure_ascii=False)

print('Done')




