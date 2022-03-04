import bert_base_cased_vocab
import object_vocab


def token2text(input_ids):
    text = [bert_base_cased_vocab.vocab[str(k)] for k in input_ids]
    return text


def label2tag(argmax_soft_labels):
    tags = [object_vocab.vocab[str(k)] for k in argmax_soft_labels]
    return tags


if __name__ == '__main__':
    input_ids = [101, 138, 14790, 2669, 8806, 1120, 170, 4059, 3482, 119,
                 102]
    text = token2text(input_ids)
    print(text)

    argmax_soft_labels = [70, 758, 287, 1470, 651, 453, 758, 1027, 255, 200, 1471, 525, 719, 719, 500, 70, 255, 1164,
                          1472, 834, 248, 906, 453, 72, 1505, 248]
    tags = label2tag(argmax_soft_labels)
    print(tags)
