#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.util import flatten, list2tuple, tuple2list


class TestDataset(Dataset):
    def __init__(self, flattened_queries):
        # flattened_queries is a list of (query, easy_ans_set, hard_ans_set, query_structure) list
        self.len = len(flattened_queries)
        self.flattened_queries = flattened_queries

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.flattened_queries[idx]

    @staticmethod
    def collate_fn(flattened_queries):
        query = [_[0] for _ in flattened_queries]
        easy_ans_set = [_[1] for _ in flattened_queries]
        hard_ans_set = [_[2] for _ in flattened_queries]
        beta_name = [_[3] for _ in flattened_queries]
        return query, easy_ans_set, hard_ans_set, beta_name


class TrainDataset(Dataset):
    def __init__(self, flattened_queries):
        # flattened_queries is a list of (query, ans_set, query_structure) list
        self.len = len(flattened_queries)
        self.flattened_queries = flattened_queries

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.flattened_queries[idx]

    @staticmethod
    def collate_fn(flattened_queries):
        query = [_[0] for _ in flattened_queries]
        ans_set = [_[1] for _ in flattened_queries]
        beta_name = [_[2] for _ in flattened_queries]
        return query, ans_set, beta_name


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
