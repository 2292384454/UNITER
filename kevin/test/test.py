import numpy as np
import bert_base_cased_vocab
import random

if __name__ == '__main__':
    input_ids = [116, 1852, 163, 4856, 896, 56354, 6536, 4523, 55, 66, 55, 163]
    no_repeat = [55, 116, 163, 896]
    tar_token = random.sample(no_repeat, 1)[0]
    need_mask_idx = np.where(np.array(input_ids) == tar_token)[0]
    print('tar_token', tar_token)
    print('need_mask_idx', need_mask_idx)
