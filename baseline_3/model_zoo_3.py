import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

hidden_size = 768

class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)

        self.bert = BertModel(config)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=5)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=5)

        self.apply(self.init_bert_weights)

    def forward(self, x_ids, x_segments, x_mask):
        x1_encoder_layers, _ = self.bert(x_ids, x_segments, x_mask, output_all_encoded_layers=False)
        ps1 = self.linear1(x1_encoder_layers)
        ps2 = self.linear2(x1_encoder_layers)

        return ps1, ps2

