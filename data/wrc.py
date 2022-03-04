"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
import random

import numpy as np
import torch
from toolz.sandbox import unzip
from torch.nn.utils.rnn import pad_sequence

from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .word_region_util import obj2bert


# 获取对应关系
def _get_mapping(converted_argmax, input_ids_list):
    mapping = []
    for word_list in converted_argmax:
        # 对于每一个 word_list ，判断 input_ids_list 中是否有 token 在其中出现，如果有就将该 token 计入 mapping 对应的位置，否则计入0
        i = 0
        while i < len(input_ids_list):
            if input_ids_list[i] in word_list:
                mapping.append(input_ids_list[i])
                break
            i = i + 1
        if i == len(input_ids_list):
            mapping.append(0)
    return mapping


class WrcDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        super().__init__(txt_db, img_db)

    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_pos_feat = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_pos_feat, img_soft_label, num_bb

    def sample_negative(self, img_fname, mapping):
        # 对 mapping 去重
        temp_set = set(mapping)
        temp_set.discard(0)
        no_repeat = list(set(temp_set))

        """ random and retry """
        # 获取原图的 img_feat, img_pos_feat
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(img_fname)
        # 选择要替换哪个 token 对应的 region
        target_token = random.sample(no_repeat, 1)[0]
        # 获取要替换的 token 对应的 region 的索引
        idxs = np.where(np.array(mapping) == target_token)[0]

        # 选择从哪张图片里选取替换 region
        target_fname = random.sample(self.all_imgs, 1)
        while target_fname == img_fname:
            target_fname = random.sample(self.all_imgs, 1)
        # 获取目标图片的 img_feat, img_pos_feat
        tar_feat, tar_pos_feat, tar_soft_labels, tar_bb = self._get_img_feat(target_fname[0])
        # 从目标图片中选择一个目标 region 用作替换
        tar_index = random.sample([i for i in range(tar_bb)], 1)[0]

        # 对所有需要替换的 region 进行替换
        for i in idxs:
            img_feat[i] = tar_feat[tar_index]
            img_pos_feat[i] = tar_pos_feat[tar_index]

        return img_feat, img_pos_feat

    def __getitem__(self, i):
        example = super().__getitem__(i)

        img_fname = example['img_fname']
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = torch.Tensor(1).long()
        argmax_soft_labels = torch.argmax(img_soft_labels[:, 1:-1], dim=1).tolist()

        # 将实体标签转化为单词 token（每一个实体标签可能对应多个单词 token）
        converted_argmax = [obj2bert[obj_label] if obj_label in obj2bert.keys() else [-1] for obj_label in
                            argmax_soft_labels]
        # 如果有交集就生成负样本
        input_ids_list = input_ids.int().tolist()
        mapping = _get_mapping(converted_argmax, input_ids_list)

        # # 加一个随机数决定是否要替换成负样本
        need_to_replace = random.randint(0, 1)

        # 如果图片和文本描述的实体检测到了交集，并且随机数决定需要替换，那么就生成负样本
        if sum(mapping) != 0 and need_to_replace == 1:
            # if sum(mapping) != 0:
            img_feat, img_pos_feat = self.sample_negative(img_fname, mapping)
            target.data.fill_(0)
        else:
            target.data.fill_(1)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([101, 138, 176, 5132, 15475, 2288, 2041, 1397,  1106, 170, 1353, 2780, 119, 102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'A', 'g', '##ira', '##ffe', 'standing', 'alone', 'next', 'to', 'a', 'small', 'tree',
        #           '.', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[1.4600e-01, 0.0000e+00, 1.6084e+00,  ..., 8.5156e+00, 8.9722e-02, 1.8613e+00],
        #                    [0.0000e+00, 0.0000e+00, 2.0762e+00,  ..., 1.0188e+01, 8.2275e-02, 2.5176e+00],
        #                    [6.9238e-01, 0.0000e+00, 5.3749e-03,  ..., 2.0508e-01, 8.1250e-01, 5.3406e-03],
        #                    ...,
        #                    [1.5781e+00, 0.0000e+00, 6.5137e-01,  ..., 5.0586e-01, 2.9883e+00, 5.5511e-02],
        #                    [0.0000e+00, 2.3108e-01, 4.6875e+00,  ..., 5.2223e-03, 9.1602e-01, 6.9275e-02],
        #                    [0.0000e+00, 3.6523e-01, 2.6309e+00,  ..., 0.0000e+00, 2.0154e-01, 2.9321e-01]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  tensor([[0.2135, 0.2571, 0.6631, 0.7266, 0.4495, 0.4695, 0.2110],
        #                        [0.4939, 0.4700, 0.9985, 0.9985, 0.5049, 0.5283, 0.2667],
        #                        [0.2908, 0.3367, 0.3330, 0.4937, 0.0422, 0.1570, 0.0066],
        #                        [0.4382, 0.2959, 0.5776, 0.4497, 0.1395, 0.1538, 0.0215],
        #                        [0.5122, 0.6362, 0.5640, 0.6880, 0.0516, 0.0517, 0.0027],
        #                        [0.3550, 0.6016, 0.3804, 0.6440, 0.0253, 0.0426, 0.0011],
        #                        [0.8901, 0.4844, 0.9985, 0.5732, 0.1088, 0.0889, 0.0097],
        #                        [0.2377, 0.8599, 0.3142, 0.9683, 0.0767, 0.1084, 0.0083],
        #                        [0.5806, 0.5571, 0.7134, 0.7090, 0.1327, 0.1517, 0.0201],
        #                        [0.6890, 0.3921, 0.8418, 0.5293, 0.1527, 0.1375, 0.0210],
        #                        [0.2595, 0.3818, 0.2959, 0.4946, 0.0363, 0.1127, 0.0041],
        #                        [0.7100, 0.0000, 0.8076, 0.1004, 0.0977, 0.1004, 0.0098],
        #                        [0.6206, 0.5298, 0.6465, 0.5732, 0.0261, 0.0434, 0.0011],
        #                        [0.6494, 0.0021, 0.9116, 0.5103, 0.2622, 0.5083, 0.1333],
        #                        [0.1108, 0.3037, 0.1724, 0.3564, 0.0616, 0.0528, 0.0033],
        #                        [0.0000, 0.2474, 0.2756, 0.9985, 0.2756, 0.7510, 0.2070],
        #                        [0.0698, 0.0000, 0.7319, 0.4592, 0.6621, 0.4592, 0.3041],
        #                        [0.7300, 0.5605, 0.9766, 0.7002, 0.2462, 0.1398, 0.0344],
        #                        [0.4807, 0.7764, 0.5356, 0.9482, 0.0547, 0.1719, 0.0094],
        #                        [0.8105, 0.7500, 0.9570, 0.8311, 0.1461, 0.0811, 0.0118],
        #                        [0.1779, 0.6372, 0.2349, 0.7402, 0.0570, 0.1028, 0.0059],
        #                        [0.7583, 0.2155, 0.8135, 0.2634, 0.0548, 0.0481, 0.0026],
        #                        [0.2241, 0.5620, 0.8652, 0.9985, 0.6411, 0.4363, 0.2797],
        #                        [0.4941, 0.5054, 0.6216, 0.5952, 0.1276, 0.0900, 0.0115],
        #                        [0.0455, 0.6235, 0.8931, 0.9985, 0.8472, 0.3748, 0.3175],
        #                        [0.1823, 0.7021, 0.2260, 0.7979, 0.0437, 0.0956, 0.0042],
        #                        [0.5981, 0.2471, 0.6328, 0.3005, 0.0346, 0.0534, 0.0018],
        #                        [0.2607, 0.0667, 0.4033, 0.3262, 0.1428, 0.2595, 0.0371],
        #                        [0.0000, 0.0283, 0.2588, 0.4841, 0.2588, 0.4558, 0.1180],
        #                        [0.7261, 0.1017, 0.8379, 0.4106, 0.1117, 0.3088, 0.0345],
        #                        [0.6587, 0.8647, 0.7207, 0.9702, 0.0618, 0.1055, 0.0065],
        #                        [0.2671, 0.5991, 0.3174, 0.7778, 0.0501, 0.1787, 0.0090],
        #                        [0.4778, 0.4683, 0.5059, 0.5122, 0.0278, 0.0439, 0.0012],
        #                        [0.7417, 0.2505, 0.8159, 0.3394, 0.0739, 0.0887, 0.0066],
        #                        [0.1036, 0.2839, 0.1682, 0.3408, 0.0646, 0.0569, 0.0037],
        #                        [0.1018, 0.2651, 0.1698, 0.3188, 0.0681, 0.0537, 0.0037],
        #                        [0.4158, 0.4937, 0.6470, 0.6084, 0.2312, 0.1151, 0.0266],
        #                        [0.0000, 0.0000, 0.3862, 0.2698, 0.3862, 0.2698, 0.1042],
        #                        [0.1144, 0.1935, 0.9824, 0.8896, 0.8682, 0.6963, 0.6045],
        #                        [0.7554, 0.2184, 0.8145, 0.2874, 0.0591, 0.0689, 0.0041],
        #                        [0.7974, 0.7485, 0.9082, 0.8315, 0.1107, 0.0833, 0.0092],
        #                        [0.4258, 0.0000, 0.9985, 0.5708, 0.5728, 0.5708, 0.3269],
        #                        [0.6934, 0.0000, 0.8208, 0.1375, 0.1276, 0.1375, 0.0175],
        #                        [0.4758, 0.4441, 0.5396, 0.4863, 0.0636, 0.0421, 0.0027],
        #                        [0.1831, 0.6431, 0.2291, 0.7637, 0.0461, 0.1207, 0.0056],
        #                        [0.0000, 0.7095, 0.0351, 0.7720, 0.0351, 0.0626, 0.0022],
        #                        [0.0000, 0.3962, 0.1270, 0.4661, 0.1270, 0.0698, 0.0089],
        #                        [0.0000, 0.5239, 0.2593, 0.9985, 0.2593, 0.4744, 0.1230],
        #                        [0.0000, 0.0000, 0.4468, 0.1986, 0.4468, 0.1986, 0.0887],
        #                        [0.7529, 0.2781, 0.8154, 0.3374, 0.0625, 0.0594, 0.0037],
        #                        [0.7231, 0.3096, 0.8120, 0.3586, 0.0889, 0.0491, 0.0044]])
        print('attn_masks: ', attn_masks)
        # attn_masks:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1])
        print('target: ', target)
        # target:  tensor([0])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def wrc_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch
