import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from baseline_3.crf import ConditionalRandomField, allowed_transitions
from baseline_3.utils import seq_len_to_mask
from configuration.dic import tag_list


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config, num_classes, encoding_type='bio', target_vocab=None, dropout=0.2):
        super(SubjectModel, self).__init__(config)

        self.bert = BertModel(config)

        self.apply(self.init_bert_weights)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.hidden_size, num_classes)

        trans = None
        if target_vocab is not None and encoding_type is not None:
            trans = allowed_transitions(target_vocab, encoding_type=encoding_type, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, x_ids, seq_len=None, target=None, max_len=None):
        feats, _ = self.bert(x_ids, output_all_encoded_layers=False)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len, max_len=max_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return pred
        else:
            loss = self.crf(logits, target, mask).mean()
            return loss

    def forward(self, x_ids, seq_len, target, max_len):
        return self._forward(x_ids, seq_len, target, max_len)

    def predict(self, x_ids, seq_len):
        return self._forward(x_ids, seq_len)




