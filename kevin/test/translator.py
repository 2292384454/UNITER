import bert_base_cased_vocab
import object_vocab


def token2text(input_ids):
    text = [bert_base_cased_vocab.vocab[str(k)] for k in input_ids]
    return text


def label2tag(argmax_soft_labels):
    tags = [object_vocab.vocab[str(k)] for k in argmax_soft_labels]
    return tags


if __name__ == '__main__':
    input_ids = [1952, 7072, 2884, 17180, 4580, 8171, 2095, 2928, 7961, 4569, 4282, 7996, 2269, 2526]
    text = token2text(input_ids)
    print(text)

    argmax_soft_labels = [374, 374, 719, 374, 374, 719, 701, 627, 1038, 374, 374, 627, 800, 106, 1261, 374, 1038, 986, 106, 914, 248]
    tags = label2tag(argmax_soft_labels)
    print(tags)
