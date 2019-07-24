import re
from pathlib import Path

import torch
from pytorch_pretrained_bert import BertConfig

from baseline.model_zoo import SubjectModel
from baseline.vocab import bert_vocab
from configuration.config import data_dir
import numpy as np

from configuration.dic import trans, trans_list
device = torch.device('cpu')

config_path = Path(data_dir)/'subject_model_config.json'
model_path = Path(data_dir)/'subject_model.pt'

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


class ExtractProcessor:
    def __init__(self):
        config = BertConfig(str(config_path))
        self.model = SubjectModel(config)
        self.model.load_state_dict(state_dict=torch.load(model_path, map_location='cpu'))
        self.bert_vocab = bert_vocab
        self.model.eval()

    def process(self, text):
        x_ids = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
        x_mask = [1] * len(x_ids)

        x_ids = torch.tensor([x_ids], dtype=torch.long, device=device)
        x_mask = torch.tensor([x_mask], dtype=torch.long, device=device)
        x_seg = torch.zeros(*x_ids.size(), dtype=torch.long, device=device)

        with torch.no_grad():
            k = self.model(x_ids, x_mask, x_seg)
            k = torch.softmax(k, dim=-1)
            k = k[0,:].detach().cpu().numpy()

        nodes = [dict(zip(map(str, range(9)), _k)) for _k in k]
        tags = viterbi(nodes)
        result = []
        for ts in re.finditer('(12+)|(34+)|(56+)|(78+)', tags):
            r = text[ts.start(): ts.end()]
            r = ''.join(r)
            result.append((r, str(ts.start()), trans_list[int(ts.group()[0])].split('_')[-1]))

        return result








