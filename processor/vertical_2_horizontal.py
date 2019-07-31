import json
from pathlib import Path

from configuration.config import data_dir


def v2h(data):
    T = []
    for l in data:
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
    return T


# fn = 'ner_total_99w.txt'
# fn = 'train_ccks2019.json'
fn = 'test_2w.json'
fn = 'train_2w.json'

j = v2h(json.load((Path(data_dir)/fn).open()))

json.dump(j, (Path(data_dir)/fn.replace('.json', '__1.json')).open('w'), ensure_ascii=False, indent=4)





