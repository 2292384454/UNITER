import numpy as np
import torch

import bert_base_cased_vocab
import random
import torch.nn.functional as F

if __name__ == '__main__':
    words_embeddings = torch.arange(0, 250, 2).unsqueeze(0).unsqueeze(0).resize(5, 5, 5)
    transformed_im = torch.arange(-250, 0, 2).unsqueeze(0).unsqueeze(0).resize(5, 5, 5)
    print(words_embeddings)
    print(transformed_im)
    word_region_maps = [{0: 2}, {1: 4}, {2: 3}, {0: 4}, {1: 2, 3: 4}]

    tmp = words_embeddings.clone()
    # 进行交换
    for i, idx_map in enumerate(word_region_maps):
        if idx_map is not None:
            txt_index = torch.LongTensor(list(idx_map.keys()))
            img_index = torch.LongTensor(list(idx_map.values()))
            words_embeddings[i][txt_index] = transformed_im[i][img_index]
            transformed_im[i][img_index] = tmp[i][txt_index]

    print(words_embeddings)
    print(transformed_im)
