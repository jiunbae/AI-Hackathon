import os
import numpy as np
from konlpy.tag import Twitter
from sklearn.model_selection import train_test_split

def preproc(sents, model, max_len):
    def factory_tagger(twitter):
        def tagger(sent):
            return [word for word, tag in twitter.pos(sent) if tag not in ['Punctuation', 'Unknown']]
        return tagger

    def factory_vectorize(model, tagger):
        def vectorize(sent):
            return model.wv[filter(lambda x: x in model.wv.vocab, tagger(sent))]
        return vectorize

    def factory_padding(max_len):
        def padder(sent):
            if len(sent) > max_len:
                return sent[:max_len]
            else:
                return np.vstack((sent, np.zeros((max_len - sent.shape[0], sent.shape[1]))))
        return padder

    tagger = factory_tagger(Twitter())
    vectorizer = factory_vectorize(model, tagger)
    padder = factory_padding(max_len)

    dat = np.array([x.strip().split('\t') for x in sents])
    x1 = dat[:, 0]
    x2 = dat[:, 1]

    vec1 = map(vectorizer, x1)
    res1 = list(map(padder, vec1))

    vec2 = map(vectorizer, x2)
    res2 = list(map(padder, vec2))

    return np.dstack((res1, res2))
    

class MovieDatset(object):
    def __init__(self, dataset_path, model, max_len):
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')

        with open(data_review, 'rt', encoding='utf-8') as f:
            self.data = preproc(f.readlines(), model, max_len)

        with open(data_label) as f:
            self.label = list(map(int, f.readlines()))

        self.train_data, self.val_data, self.train_label, self.val_label = \
            train_test_split(self.data, self.label, test_size=0.3, random_state=42)

    def get_batcher(self, batch_size):
        return Batch(self.train_data, self.train_label, batch_size)


class Batch(object):
    def __init__(self, x, y, batch_size):
        self.total_x = x
        self.total_y = y
        self.batch_size = batch_size

        self.iter_per_epoch = len(x) // batch_size
        self.epochs_completed = 0

        self._iter = 0

    def __call__(self):
        start = self._iter * self.batch_size
        end = (self._iter + 1) * self.batch_size

        batch_x = self.total_x[start:end]
        batch_y = self.total_y[start:end]

        self._iter += 1
        if self._iter == self.iter_per_epoch:
            self.epochs_completed += 1
            self._iter = 0

        return batch_x, batch_y