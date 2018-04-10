# -*- coding: utf-8 -*-

import os

import numpy as np

from process import preprocess

class KinQueryDataset:
    def __init__(self, dataset_path: str, max_length: int):
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = preprocess(f.readlines(), max_length)
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]
