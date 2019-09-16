import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_pretrained_bert import BertConfig

from baseline import device
from baseline.vocab import bert_vocab
from baseline_3.sequence_labeling import SubjectModel
from configuration.config import data_dir
from configuration.dic import trans, tag_list, tag_dictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

num_class = len(tag_dictionary)
model_dir = 'model_crf'

config_path = Path(data_dir)/'subject_model_config.json'
model_path = Path(data_dir)/model_dir/'subject_model.pt'

zy = {i:trans[i] for i in trans}

# def viterbi(nodes):
#     paths = nodes[0]
#     for l in range(1,len(nodes)):
#         paths_ = paths.copy()
#         paths = {}
#         for i in nodes[l].keys():
#             nows = {}
#             for j in paths_.keys():
#                 if j[-1]+i in zy.keys():
#                     nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
#             k = np.argmax(list(nows.values()))
#             paths[list(nows.keys())[k]] = list(nows.values())[k]
#     return list(paths.keys())[np.argmax(list(paths.values()))]

config = BertConfig(str(config_path))
model = SubjectModel(config,num_classes=num_class, target_vocab=dict(zip(range(len(tag_list)), tag_list)))
model.load_state_dict(state_dict=torch.load(model_path, map_location='cpu'if not torch.cuda.is_available() else None))
model.to(device)
model.eval()

bert_vocab = bert_vocab

dev_data = json.load((Path(data_dir)/'test_0729.json').open())
A, B, C = 1e-10, 1e-10, 1e-10
err_dict = defaultdict(list)

cat_dict = defaultdict(lambda: 1e-10)
for eval_idx, d in enumerate(dev_data):
    text, mention = d
    text = ''.join(text)

    T = []
    B_idx = [i for i, l in enumerate(mention) if 'B' in l]
    for i in B_idx:
        e = text[i]
        e_n = mention[i]
        j = i + 1
        while j < len(mention):
            if mention[j] == 'O' or 'B' in mention[j]:
                break
            e += text[j]
            j += 1
        T.append((e, str(i), e_n.split('_')[-1]))
        # T.append((e, str(i)))

    with torch.no_grad():
        _X = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
        max_len = len(_X)

        _X_Len = torch.tensor([len(_X)], dtype=torch.long, device=device)
        _X = torch.tensor([_X], dtype=torch.long, device=device)

        pred_tags = model(_X, _X_Len)[0]
        pred_tags = [tag_list[_] for _ in pred_tags]

    R = []
    pred_B_idx = [i for i, l in enumerate(pred_tags) if 'B' in l]
    for i in pred_B_idx:
        e = text[i]
        e_n = pred_tags[i]
        j = i + 1
        while j < len(pred_tags):
            if pred_tags[j] == 'O' or 'B' in pred_tags[j]:
                break
            e += text[j]
            j += 1
        R.append((e, str(i), e_n.split('_')[-1]))

    if '体检' in text:
        R.append(('体检',str(text.index('体检')),'diagnosis'))

    R = set(R)
    T = set(T)

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
json.dump(err_dict, Path(f'err_log_{model_dir}__[el_pt_subject.py].json').open('w'), ensure_ascii=False, indent=4)


for cat in ['disease', 'drug', 'diagnosis', 'symptom']:
    logger.info(f'cate:{cat} - '
                f'precision:{cat_dict[cat+"_A"] / cat_dict[cat+"_B"]:.5f} - '
                f'recall:{cat_dict[cat+"_A"] / cat_dict[cat+"_C"]:.5f} - '
                f'f1:{2*cat_dict[cat+"_A"] / (cat_dict[cat+"_B"]+cat_dict[cat+"_C"]):.5f}')





