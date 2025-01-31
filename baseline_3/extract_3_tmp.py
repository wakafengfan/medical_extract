import collections
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertAdam

from baseline import device
from baseline_3.sequence_labeling import SubjectModel
from configuration.config import data_dir, bert_vocab_path, bert_data_path, bert_model_path
from configuration.dic import tag_dictionary, tag_list

hidden_size = 768
epoch_num = 10
batch_size = 32
num_class = len(tag_dictionary)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

train_data = json.load((Path(data_dir)/'train_0729.json').open())
dev_data = json.load((Path(data_dir)/'test_0729.json').open())


def seq_padding(X):
    ML = max(map(len, X))
    return np.array([list(x) + [0] * (ML - len(x)) for x in X])


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


bert_vocab = load_vocab(bert_vocab_path)

# wv_model = gensim.models.KeyedVectors.load(str(Path(data_dir) / 'tencent_embed_for_el2019'))
# word2vec = wv_model.wv.syn0
# word_size = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((1, word_size)), np.zeros((1, word_size)), word2vec])  # [word_size+2,200]
# id2word = {i + 2: j for i, j in enumerate(wv_model.wv.index2word)}
# word2id = {j: i for i, j in id2word.items()}
#
#
# def seq2vec(token_ids):
#     V = []
#     for s in token_ids:
#         V.append([])
#         for w in s:
#             for _ in w:
#                 V[-1].append(word2id.get(w, 1))
#     V = seq_padding(V)
#     V = word2vec[V]
#     return V


class data_generator:
    def __init__(self, data, bs=batch_size):
        self.data = data
        self.batch_size = bs
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        X, Y, L = [], [], []
        for i in idxs:
            text, label = self.data[i]

            x = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
            y = [tag_dictionary[l] for l in label]

            X.append(x)
            Y.append(y)
            L.append(len(x))

            if len(X) == self.batch_size or i == idxs[-1]:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                Y = torch.tensor(seq_padding(Y), dtype=torch.long)
                L = torch.tensor(L, dtype=torch.long)

                yield [X, Y, L, max(map(len, X))]

                X, Y, L = [], [], []


subject_model = SubjectModel.from_pretrained(pretrained_model_name_or_path=bert_model_path,
                                             cache_dir=bert_data_path,
                                             num_classes=num_class,
                                             target_vocab=dict(zip(range(len(tag_list)), tag_list)))

subject_model.to(device)

n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    torch.cuda.manual_seed_all(42)

    logger.info(f'let us use {n_gpu} gpu')
    subject_model = nn.DataParallel(subject_model)


# optim
param_optimizer = list(subject_model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

learning_rate = 5e-5
warmup_proportion = 0.1
num_train_optimization_steps = len(train_data) // batch_size * epoch_num
logger.info(f'num_train_optimization: {num_train_optimization_steps}')

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)


# zy = {i:np.log(trans[i]) for i in trans}
#
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
#
# def extract_items(text_in):
#     _X = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text_in]
#     _X_MASK = [1] * len(_X)
#     _X = torch.tensor([_X], dtype=torch.long, device=device)
#     _X_MASK = torch.tensor([_X_MASK], dtype=torch.long, device=device)
#     _X_SEG = torch.zeros(*_X.size(), dtype=torch.long, device=device)
#
#     with torch.no_grad():
#         _k = subject_model(_X, _X_SEG, _X_MASK)
#         _k = _k[0, :].detach().cpu().numpy()
#     nodes = [dict(zip(list(map(str, range(9))), k)) for k in np.log(_k)]
#     tags = viterbi(nodes)
#     result = []
#     for ts in re.finditer('(12+)|(34+)|(56+)|(78+)', tags):
#         r = text_in[ts.start(): ts.end()]
#         r = ''.join(r)
#         result.append((r, str(ts.start()), trans_list[int(ts.group()[0])].split('_')[-1]))
#
#     return result

best_score = 0
best_epoch = 0
train_D = data_generator(train_data)
for epoch in range(epoch_num):
    subject_model.train()
    batch_idx = 0
    tr_total_loss = 0
    dev_total_loss = 0

    for batch in train_D:
        batch_idx += 1
        batch = tuple(t.to(device) if i<len(batch)-1 else t for i,t in enumerate(batch))
        X, Y, L, max_len = batch
        loss = subject_model(X, L, Y, max_len)

        if n_gpu > 1:
            loss = loss.mean()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tr_total_loss += loss.item()
        if batch_idx % 100 == 0:
            logger.info(f'Epoch:{epoch} - batch:{batch_idx}/{train_D.steps} - loss: {tr_total_loss / batch_idx:.8f}')

    subject_model.eval()
    A, B, C = 1e-10, 1e-10, 1e-10
    err_dict = defaultdict(list)
    for eval_idx, d in enumerate(dev_data):
        tt, ll = d
        with torch.no_grad():
            _X = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in tt]
            max_len = len(_X)

            _X_Len = torch.tensor([len(_X)], dtype=torch.long, device=device)
            _X = torch.tensor([_X], dtype=torch.long, device=device)

            pred_tags = subject_model(_X, _X_Len)[0]
            pred_tags = [tag_list[_] for _ in pred_tags]

        R = []
        pred_B_idx = [i for i, l in enumerate(pred_tags) if 'B' in l]
        for i in pred_B_idx:
            e = tt[i]
            e_n = pred_tags[i]
            j = i+1
            while j<len(pred_tags):
                if pred_tags[j]=='O' or 'B' in pred_tags[j]:
                    break
                e += tt[j]
                j += 1
            R.append((e,str(i),e_n.split('_')[-1]))

        T = []
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
            T.append((e, str(i), e_n.split('_')[-1]))

        R = set(R)
        T = set(T)

        A += len(R & T)
        B += len(R)
        C += len(T)

        if R != T:
            err_dict['err'].append({'text': ''.join(tt),
                                    'tags': ll,
                                    'mention_data': list(T),
                                    'predict': list(R)})
        if eval_idx % 100 == 0:
            logger.info(f'eval_idx:{eval_idx} - precision:{A/B:.5f} - recall:{A/C:.5f} - f1:{2 * A / (B + C):.5f}')

    f1, precision, recall = 2 * A / (B + C), A / B, A / C
    if f1 > best_score:
        best_score = f1
        best_epoch = epoch

        json.dump(err_dict, (Path(data_dir) / 'err_log_dev__[extract.py].json').open('w'), ensure_ascii=False)

        s_model_to_save = subject_model.module if hasattr(subject_model, 'module') else subject_model
        torch.save(s_model_to_save.state_dict(), 'subject_model.pt')

        Path('subject_model_config.json').open('w').write(s_model_to_save.config.to_json_string())

    logger.info(
        f'Epoch:{epoch}-precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f} - best f1: {best_score:.4f} - best epoch:{best_epoch}')


# config = BertConfig(str(Path(data_dir) / 'subject_model_config.json'))
# subject_model = SubjectModel(config)
# subject_model.load_state_dict(
#     torch.load(Path(data_dir) / 'subject_model.pt', map_location='cpu' if not torch.cuda.is_available() else None))
#
# subject_model.to(device)
# if n_gpu > 1:
#     torch.cuda.manual_seed_all(42)
#
#     logger.info(f'let us use {n_gpu} gpu')
#     subject_model = torch.nn.DataParallel(subject_model)
#
# subject_model.eval()
# A, B, C = 1e-10, 1e-10, 1e-10
# err_dict = defaultdict(list)
# for eval_idx, d in enumerate(test_data):
#     m_ = [m for m in d['mention_data'] if m[0] in kb2id]
#
#     R = set(map(lambda x: (str(x[0]), str(x[1])), set(extract_items(d['text']))))
#     T = set(map(lambda x: (str(x[0]), str(x[1])), set(m_)))
#     A += len(R & T)
#     B += len(R)
#     C += len(T)
#
#     if R != T:
#         err_dict['err'].append({'text': d['text'],
#                                 'mention_data': list(T),
#                                 'predict': list(R)})
#     if eval_idx % 100 == 0:
#         logger.info(f'Test eval_idx:{eval_idx} - precision:{A/B:.5f} - recall:{A/C:.5f} - f1:{2 * A / (B + C):.5f}')
#
# json.dump(err_dict, (Path(data_dir) / 'err_log_tst__[el_pt_subject.py].json').open('w'), ensure_ascii=False)
#
# f1, precision, recall = 2 * A / (B + C), A / B, A / C
# logger.info(f'Test precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f}')

