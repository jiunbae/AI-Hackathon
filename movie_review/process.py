import numpy as np

from konlpy.tag import Twitter
twt = Twitter()

from random import randint
from collections import defaultdict, Counter
from itertools import chain

def continuous(data: list, length: int = 100):
    if len(data) <= length: return data + [0] * (length - len(data))
    pos = randint(length - len(data))
    return data[pos:length]

def preprocess(data: list, minlen: int, maxlen: int) -> list:
    voc_size = 20000

    # get tokenized text
    parsed = [twt.morphs(t) for t in data if len(t)>minlen and len(t)<maxlen]

    # get common words
    commons, _ = zip(*Counter(chain(*parsed)).most_common(voc_size - 2))
    
    # generate dictionary for word -> index
    voc2idx = {v: i for i, v in enumerate(['<PAD>'] + list(commons) + ['<UNK>'])}

    # apply vectorize to parsed
    vectorized = map(lambda r: [voc2idx.get(v, voc2idx['<UNK>']) for v in r], parsed)

    return list(vectorized)
