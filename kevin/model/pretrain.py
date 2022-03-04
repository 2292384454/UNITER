"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from model.layer import GELU, BertOnlyMLMHead
from model.model import UniterModel, UniterPreTrainedModel
from model.ot import optimal_transport_dist


class RegionFeatureRegression(nn.Module):
    " for MRM"

    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"

    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.wrc_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)

        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']  # todo ？
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(input_ids, position_ids,
                                     img_feat, img_pos_feat,
                                     attention_mask, gather_index,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task.startswith('wrc'):
            targets = batch['targets']
            return self.forward_wrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)
        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('masked_output', masked_output)
        # masked_output tensor([[-0.0764, 0.0078, -0.1247, ..., 0.2786, -0.2310, 0.0988],
        #                       [0.0385, -0.1444, -0.0632, ..., -0.2288, -0.1781, -0.0041],
        #                       [-0.1689, 0.0309, -0.3494, ..., -0.2705, 0.4321, -0.3420],
        #                       ...,
        #                       [0.0401, 0.3455, -0.1455, ..., -0.4719, -0.4094, 0.0135],
        #                       [-0.3184, 0.1279, 0.1858, ..., -0.2452, 0.1707, -0.4253],
        #                       [0.1920, -0.9717, -0.6885, ..., 0.4739, -0.3340, 0.4231]],
        #                       device='cuda:0', dtype=torch.float16, grad_fn= < ViewBackward >)
        print('masked_output.size()', masked_output.size())
        # masked_output.size() torch.Size([144, 768])
        print('prediction_scores', prediction_scores)
        # prediction_scores tensor([[-8.2891, -8.6719, -8.4453, ..., -8.6172, -8.2344, -9.0703],
        #                           [-6.6719, -6.3945, -6.2344, ..., -6.1992, -6.6172, -8.1719],
        #                           [-9.5469, -9.3203, -9.7578, ..., -9.6797, -8.9688, -8.8047],
        #                           ...,
        #                           [-7.3750, -8.1172, -7.6094, ..., -7.1211, -6.0391, -7.5391],
        #                           [-5.2539, -5.3398, -5.4141, ..., -4.9492, -4.4883, -3.8203],
        #                           [-9.8828, -10.1719, -9.8281, ..., -10.2500, -10.2891, -9.1562]],
        #                           device='cuda:0', dtype=torch.float16, grad_fn= < AddBackward0 >)
        print('prediction_scores.size()', prediction_scores.size())
        # prediction_scores.size() torch.Size([144, 28996])
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    # KevinHwang 改进版
    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        """
        @parameters
            input_ids: [batch_size, seq_length], 句子中词语 id 组成的 tensor
            position_ids: [1, seq_length], 句子中词语位置组成的 tensor
            img_feat: [batch_size, num_bb, 2048], 图像中 region feature 组成的 tensor
            img_pos_feat: [batch_size, num_bb, 7], 图像中 region 位置信息组成的 tensor
            attention_mask: [batch_size, max(txt_len + img_len)]
            gather_index: [batch_size, max(txt_len + img_len)]
            targets: [batch_size], tensor of 0/1, 表示输入的文本和图片是否匹配
            ot_inputs: a dict
            compute_loss: True or False
        """

        # sequence_output: [batch_size, max(txt_len + img_len), embedding_size],每个词语和region的向量表示
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        # pooled_output:[batch_size, embedding_size] [CLS]的向量表示，可以用来代表句子和图像的联合表示
        pooled_output = self.uniter.pooler(sequence_output)

        itm_scores = self.itm_output(pooled_output)

        # ot_inputs.dict_keys(['ot_scatter', 'scatter_max', 'txt_pad', 'img_pad'])
        # ot_scatter.Size([batch_size, max(txt_len + img_len)]) torch.int64
        # scatter_max int
        # txt_pad.Size([batch_size, txt_len]) torch.bool
        # img_pad.Size([batch_size, img_len]) torch.bool

        # # OT loss
        # if ot_inputs is not None:
        #     ot_scatter = ot_inputs['ot_scatter']
        #
        #     b = sequence_output.size(0)  # 即 batch_size
        #     tl = input_ids.size(1)  # 即本批次最长的文本 tokens 的长度
        #     il = img_feat.size(1)  # 即本批次最长的图像 features 的长度
        #     max_l = max(ot_inputs['scatter_max'] + 1, tl + il)
        #
        #     # unsqueeze(-1) 对张量的最后一个维度扩展一维
        #     # expand_as(sequence_output) 把张量扩展成和 sequence_output 一样的形状
        #     ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
        #     ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
        #                           dtype=sequence_output.dtype,
        #                           device=sequence_output.device
        #                           ).scatter_(dim=1, index=ot_scatter,
        #                                      src=sequence_output)
        #     txt_emb = ctx_emb[:, :tl, :]
        #     img_emb = ctx_emb[:, tl:tl + il, :]
        #
        #     txt_pad = ot_inputs['txt_pad']
        #     img_pad = ot_inputs['img_pad']
        #     # NOTE: run in fp32 for stability
        #     ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
        #                                      txt_pad, img_pad).to(txt_emb)
        #     ot_pos_dist = ot_dist.masked_select(targets == 1)
        #     ot_neg_dist = ot_dist.masked_select(targets == 0)
        #     ot_loss = (ot_pos_dist, ot_neg_dist)
        # else:
        #     ot_loss = None

        ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            # KevinHwang: itm_loss with infoNCE
            # itm_loss = - torch.div(torch.exp(torch.mul(itm_scores[:, 0], targets.type_as(itm_scores))),
            #                        torch.exp(itm_scores[:, 0]))
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label

    def forward_wrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, compute_loss=True):

        # print("正样本比例为：", torch.sum(targets).item() / targets.size(0))

        # sequence_output: [batch_size, max(txt_len + img_len), embedding_size],每个词语和region的向量表示
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        # pooled_output:[batch_size, embedding_size] [CLS]的向量表示，可以用来代表句子和图像的联合表示
        pooled_output = self.uniter.pooler(sequence_output)

        wrc_scores = self.wrc_output(pooled_output)

        if compute_loss:
            wrc_loss = F.cross_entropy(wrc_scores, targets, reduction='none')
            # KevinHwang: itm_loss with infoNCE
            # wrc_loss = - torch.div(torch.exp(torch.mul(wrc_scores[:, 0], targets.type_as(wrc_scores))),
            #                        torch.exp(wrc_scores[:, 0]))
            return wrc_loss
        else:
            return wrc_scores
