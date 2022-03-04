import lmdb
import msgpack
import numpy as np
from lz4.frame import decompress
from PIL import Image

import object_vocab

"""
/img/coco_train2014/feat_th0.2_max100_min10
|-- coco_train2014_000000000009.npz
| |-- norm_bb
| | |-- nd
| | |-- type
| | |-- kind
| | |-- shape
| | |-- data
| |-- features
| | |-- nd
| | |-- type
| | |-- kind
| | |-- shape
| | |-- data
| |-- conf
| | |-- nd
| | |-- type
| | |-- kind
| | |-- shape
| | |-- data
| |-- soft_labels
| | |-- nd
| | |-- type
| | |-- kind
| | |-- shape
| | |-- data
|-- coco_train2014_.......npz
| |-- ......
|-- coco_train2014_.......npz
|-- ......
"""


def show_dict_keys(in_dict, num_bb, prefix=''):
    for k, v in in_dict.items():  # 迭代当前的字典层级
        k = k if isinstance(k, str) else k.decode('utf-8')
        print(prefix + k)
        # 如果当前data属于dict类型, 进行递归
        if isinstance(v, dict):
            show_dict_keys(v, num_bb, prefix=prefix + '--')
        else:
            if k == 'data':
                # print(list(bytes(v)))
                print(np.frombuffer(v, dtype=np.float16).reshape(num_bb, -1))
            else:
                print(v)


if __name__ == '__main__':
    '''
    img_env = lmdb.open("/home/hky/processed_data_and_pretrained_models/img_db/coco_train2014/feat_th0.2_max100_min10",
                        readonly=True, create=False)
    img_txn = img_env.begin()

    # 通过cursor()遍历所有数据和键值
    key = b'coco_train2014_000000204103.npz'
    value = img_txn.get(key)
    img_dump = msgpack.loads(value, raw=False)
    print('key:', key)
    num_bb = img_dump['norm_bb'][b'shape'][0]
    print('num_bb =', num_bb)
    show_dict_keys(img_dump, num_bb)
    print("the objects of each region:")
    soft_labels_data = np.frombuffer(img_dump['soft_labels'][b'data'], dtype=np.float16).reshape(num_bb, -1)[:,
                       1:-1].copy()
    conf = [k for k in np.max(soft_labels_data, axis=1)]
    soft_labels = [object_vocab.vocab[str(k)] for k in np.argmax(soft_labels_data, axis=1)]
    dict = dict(zip(conf, soft_labels))
    print(np.array(conf) - np.frombuffer(img_dump['conf'][b'data'], dtype=np.float16))
    print(dict)
    print(np.argmax(soft_labels_data, axis=1))
    norm_bb = np.frombuffer(img_dump['norm_bb'][b'data'], dtype=np.float16).reshape(num_bb, -1)
    print(norm_bb)

    img_env.close()
    # # 结果见./npz_dict

    ####################################################################################################################

    '''
    txt_env = lmdb.open("/home/hky/processed_data_and_pretrained_models/txt_db/pretrain_coco_train.db", readonly=True,
                        create=False)
    txt_txn = txt_env.begin()
    i = 0
    key = b'166386'
    value = txt_txn.get(key)
    i = i + 1
    txt_dump = msgpack.loads(decompress(value), raw=False)
    print('key:', key)
    print('txt_dump: ', txt_dump)

    txt_env.close()

    # key: b'166386'
    # txt_dump: {'id': 166386,
    #            'dataset': 'coco',
    #            'split': 'train',
    #            'sent': 'a woman in a white shirt and a white dog at the beach',
    #            'bbox': None,
    #            'dataset_image_id': 204103,
    #            'file_path': 'train/COCO_train2014_000000204103.jpg',
    #            'image_id': 204103,
    #            'toked_caption': ['a', 'woman', 'in', 'a', 'white', 'shirt', 'and', 'a', 'white', 'dog', 'at', 'the',
    #                              'beach'],
    #            'input_ids': [170, 1590, 1107, 170, 1653, 2969, 1105, 170, 1653, 3676, 1120, 1103, 4640],
    #            'img_fname': 'coco_train2014_000000204103.npz'}
