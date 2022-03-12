import numpy as np
import torch

import bert_base_cased_vocab
import random
import torch.nn.functional as F

if __name__ == '__main__':
    # batch_size=3, txt_len=4, img_len=5, emb_len=6
    txt_output = torch.tensor(np.random.normal(size=(112, 26, 768)))
    img_output = torch.tensor(np.random.normal(size=(112, 23, 768)))
    # 进行归一化
    txt_output = F.normalize(txt_output, p=2, dim=2)
    img_output = F.normalize(img_output, p=2, dim=2)
    # 对 img_output 进行转置
    img_output = torch.transpose(img_output, 1, 2)

    mat = torch.bmm(txt_output, img_output).flatten(1)

    mat_mask = torch.tensor(np.random.randint(0, 2, size=(112, 26, 23))).flatten(1)

    print('mat:', mat)
    wrc_loss = -torch.mean(F.log_softmax(mat, dim=1) * mat_mask, dim=1)
    print('wrc_loss:', wrc_loss)
