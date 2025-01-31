import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_pretrained_bert import BertConfig

from baseline import device
from baseline.model_zoo import SubjectModel
from baseline.vocab import bert_vocab
from configuration.config import data_dir
import numpy as np

from configuration.dic import trans, tag_list

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dir = 'model_1'
config_path = Path(data_dir)/model_dir/'subject_model_config.json'
model_path = Path(data_dir)/model_dir/'subject_model.pt'

zy = {i:trans[i] for i in trans}

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(list(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(list(paths.values()))]

config = BertConfig(str(config_path))
model = SubjectModel(config)
model.load_state_dict(state_dict=torch.load(model_path, map_location='cpu'if not torch.cuda.is_available() else None))
model.to(device)
model.eval()

bert_vocab = bert_vocab

dev_data = json.load((Path(data_dir)/'ner_v2.json').open())
A, B, C = 1e-10, 1e-10, 1e-10
cat_dict = defaultdict(lambda: 1e-10)

err_dict = defaultdict(list)
for eval_idx, d in enumerate(dev_data):
    text = d['text']
    mention = d['mention']

    if len(text) > 500:
        continue

    x_ids = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
    x_mask = [1] * len(x_ids)

    x_ids = torch.tensor([x_ids], dtype=torch.long, device=device)
    x_mask = torch.tensor([x_mask], dtype=torch.long, device=device)
    x_seg = torch.zeros(*x_ids.size(), dtype=torch.long, device=device)

    with torch.no_grad():
        try:

            k = model(x_ids, x_mask, x_seg)
            k = torch.softmax(k, dim=-1)
            kk = k[0,:].detach().cpu().numpy()
        except Exception:
            pass

    nodes = [dict(zip(map(str, range(9)), _k)) for _k in kk]
    tags = viterbi(nodes)
    R = []
    for ts in re.finditer('(12+)|(34+)|(56+)|(78+)', tags):
        r = text[ts.start(): ts.end()]
        r = ''.join(r)
        R.append((r, str(ts.start()), tag_list[int(ts.group()[0])].split('_')[-1]))

    R = set(R)
    T = set(tuple(m) for m in mention)

    A += len(R & T)
    B += len(R)
    C += len(T)

    for cat in ['disease', 'drug', 'diagnosis', 'symptom']:
        R_ = set(r for r in R if r[2] == cat)
        T_ = set(t for t in T if t[2] == cat)
        cat_dict[f'{cat}_A'] += len(R_ & T_)
        cat_dict[f'{cat}_B'] += len(R_)
        cat_dict[f'{cat}_C'] += len(T_)

    if R != T:
        err_dict['err'].append({'text': text,
                                'mention_data': list(T),
                                'predict': list(R)})
    if eval_idx % 100 == 0:
        logger.info(f'eval_idx:{eval_idx} - precision:{A / B:.5f} - recall:{A / C:.5f} - f1:{2 * A / (B + C):.5f}')
        for cat in ['disease', 'drug', 'diagnosis', 'symptom']:
            logger.info(f'cate:{cat} - '
                        f'precision:{cat_dict[cat + "_A"] / cat_dict[cat + "_B"]:.5f} - '
                        f'recall:{cat_dict[cat + "_A"] / cat_dict[cat + "_C"]:.5f} - '
                        f'f1:{2 * cat_dict[cat + "_A"] / (cat_dict[cat + "_B"] + cat_dict[cat + "_C"]):.5f}')
        logger.info(f'\n')

f1, precision, recall = 2 * A / (B + C), A / B, A / C
logger.info(f'precision:{A / B:.5f} - recall:{A / C:.5f} - f1:{2 * A / (B + C):.5f}')
json.dump(err_dict, Path('err_log_tst__[el_pt_subject.py].json').open('w'), ensure_ascii=False, indent=4)








