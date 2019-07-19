import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

hidden_size = 768


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)

        self.bert = BertModel(config)
        self.linear = nn.Linear(in_features=hidden_size, out_features=9)

        self.apply(self.init_bert_weights)

    def forward(self, x_ids, x_segments, x_mask):
        x1_encoder_layers, _ = self.bert(x_ids, x_segments, x_mask, output_all_encoded_layers=False)
        ps = F.softmax(self.linear(x1_encoder_layers), dim=-1)

        return ps