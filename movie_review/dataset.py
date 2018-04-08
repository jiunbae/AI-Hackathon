# -*- coding: utf-8 -*-

from os import path

import numpy as np
from torch.utils.data import Dataset

from process import preprocess

class MovieReviewDataset(Dataset):
    def __init__(self, dataset_path: str, minlen: int, maxlen: int):
        data_review = path.join(dataset_path, 'train', 'train_data')
        data_label = path.join(dataset_path, 'train', 'train_label')

        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews = preprocess(f.readlines(), minlen, maxlen)

        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx: int):
        return self.reviews[idx], self.labels[idx]
