from pathlib import Path

import torch
from pytorch_pretrained_bert import BertConfig

from baseline.vocab import bert_vocab
from baseline_3.sequence_labeling import SubjectModel
from configuration.config import data_dir
from configuration.dic import trans, tag_list, tag_dictionary

device = torch.device('cpu')


num_class = len(tag_dictionary)
model_dir = 'model_crf'
config_path = Path(data_dir)/'subject_model_config.json'
model_path = Path(data_dir)/model_dir/'subject_model.pt'

zy = {i:trans[i] for i in trans}


class ExtractProcessor:
    def __init__(self):
        config = BertConfig(str(config_path))
        self.model = SubjectModel(config,num_classes=num_class, target_vocab=dict(zip(range(len(tag_list)), tag_list)))
        self.model.load_state_dict(state_dict=torch.load(model_path, map_location='cpu'))
        self.bert_vocab = bert_vocab
        self.model.eval()

    def process(self, text):
        with torch.no_grad():
            _X = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
            max_len = len(_X)

            _X_Len = torch.tensor([len(_X)], dtype=torch.long, device=device)
            _X = torch.tensor([_X], dtype=torch.long, device=device)

            pred_tags = self.model(_X, _X_Len)[0]
            pred_tags = [tag_list[_] for _ in pred_tags]

        result = []
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
            result.append((e, str(i), e_n.split('_')[-1]))

        if '体检' in text:
            result.append(('体检', str(text.index('体检')), 'diagnosis'))

        return result








