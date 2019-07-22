import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from baseline import device
from configuration.dic import tag_dictionary, trans_list

hidden_size = 768
START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"
tagset_size = len(tag_dictionary)

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


class SubjectModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SubjectModel, self).__init__(config)

        self.bert = BertModel(config)
        self.linear = nn.Linear(in_features=hidden_size, out_features=tagset_size)

        self.transitions = torch.nn.Parameter(
            torch.randn(tagset_size, tagset_size)
        )
        self.transitions.detach()[tag_dictionary.get(START_TAG), :] = -10000
        self.transitions.detach()[:, tag_dictionary.get(STOP_TAG)] = -10000

        self.apply(self.init_bert_weights)

    def forward(self, x_ids, x_segments, x_mask):
        x1_encoder_layers, _ = self.bert(x_ids, x_segments, x_mask, output_all_encoded_layers=False)
        ps = F.softmax(self.linear(x1_encoder_layers), dim=-1)

        return ps

    def _score_sentence(self, feats, tags, lens_):  # tags: [b,s]

        start = torch.tensor(
            [tag_dictionary.get(START_TAG)], device=device
        )  # [1]
        start = start[None, :].repeat(tags.shape[0], 1)  # [b,1]

        stop = torch.tensor(
            [tag_dictionary.get(STOP_TAG)], device=device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)   # pad_stop_tags: [b,s+1]

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = tag_dictionary.get(STOP_TAG)

        score = torch.FloatTensor(feats.shape[0]).to(device)  # score: [b]

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score

    def calculate_loss(self, features, lengths, tags):

        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)

        score = forward_score - gold_score

        return score.mean()


    def obtain_labels(self, feature, lengths):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        # sentences.sort(key=lambda x: len(x), reverse=True)
        #
        # lengths = [len(c for c in sentence) for sentence in sentences]

        tags = []
        all_tags = []
        for feats, length in zip(feature, lengths):
            confidences, tag_seq, scores = self.viterbi_decode(feats[:length])

            tags.append([trans_list[tag] for tag in tag_seq])

            # all_tags.append([[trans_list[torch.tensor(score_id).detach().item()] for score_id, score in enumerate(score_dist)] for score_dist in scores])

        return tags

    def viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []

        init_vvars = (
            torch.FloatTensor(1, tagset_size).to(device).fill_(-10000.0)
        )
        init_vvars[0][tag_dictionary.get(START_TAG)] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                forward_var.view(1, -1).expand(tagset_size, tagset_size)
                + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
            forward_var
            + self.transitions[tag_dictionary.get(STOP_TAG)]
        )
        terminal_var.detach()[tag_dictionary.get(STOP_TAG)] = -10000.0
        terminal_var.detach()[
            tag_dictionary.get(START_TAG)
        ] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])
        # This has been taken from https://github.com/zalandoresearch/flair/pull/642
        swap_best_path, swap_max_score = (
            best_path[0],
            scores[-1].index(max(scores[-1])),
        )
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )

        start = best_path.pop()
        assert start == tag_dictionary.get(START_TAG)
        best_path.reverse()
        return best_scores, best_path, scores

    def _forward_alg(self, feats, lens_):  # [b,s,tag_size]

        init_alphas = torch.FloatTensor(tagset_size).fill_(-10000.0)  # [tag_size]
        init_alphas[tag_dictionary.get(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=device,
        )  # [b,s+1,tag_size]

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)  # [b,tag_size]

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)  # [b,tag_size,tag_size]  # 从当前step转向下个tag的trans_score

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]  # [b,tag_size] 当前step的emit_score,不考虑之前tag

            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])  # [b,tag_size,tag_size] 每个tag重复tag_size个
                + transitions
                + forward_var[:, i, :][:, :, None]  # [b,tag_size,tag_size]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)  # [b,tag_size]

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))  # [b,]

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
            tag_dictionary.get(STOP_TAG)
        ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha