#!/usr/bin/python3

import os
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from fol import parse_foq_formula


class Task:
    def __init__(self, filename):
        self.filename = filename
        self.query_instance = None
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.i = 0
        self.length = 0
        self.idxlist = []
        self._load()

    def _load(self):
        dense = self.filename.replace('csv', 'pickle')
        if os.path.exists(dense):
            print("load from existed files")
            with open(dense, 'rb') as f:
                data = pickle.load(f)
                self.query_instance = data['query_instance']
                self.answer_set = data['answer_set']
                self.easy_answer_set = data['easy_answer_set']
                self.hard_answer_set = data['hard_answer_set']
                self.length = len(self.query_instance)
        else:
            df = pd.read_csv(self.filename)
            self._parse(df)
            data = {}
            data['query_instance'] = self.query_instance
            data['answer_set'] = self.answer_set
            data['easy_answer_set'] = self.easy_answer_set
            data['hard_answer_set'] = self.hard_answer_set
            try:
                with open(dense, 'wb') as f:
                    pickle.dump(data, f)
            except:
                print("no right to save here")

    def __len__(self):
        return self.length

    def setup_iteration(self):
        self.i = 0
        self.idxlist = np.random.permutation(len(self))

    def batch_estimation(self, estimator, batch_size):
        batch_indices = self.idxlist[self.i, self.i+batch_size]
        self.i += batch_size
        batch_embedding = self.query_instance.embedding_estimation(
            estimator=estimator, 
            batch_indices=batch_indices)
        return batch_embedding, batch_indices

    def _parse(self, df):
        for q in tqdm(df['query']):
            if self.query_instance is None:
                self.query_instance = parse_foq_formula(q)
            else:
                self.query_instance.additive_ground(q)

        if 'answer_set' in df.columns:
            self.answer_set = df.answer_set.map(lambda x: eval(x)).tolist()
            assert len(self.query_instance) == len(self.answer_set)

        if 'easy_answer_set' in df.columns:
            self.easy_answer_set = df.easy_answer_set.map(
                lambda x: eval(x)).tolist()
            assert len(self.query_instance) == len(self.easy_answer_set)

        if 'hard_answer_set' in df.columns:
            self.hard_answer_set = df.hard_answer_set.map(
                lambda x: eval(x)).tolist()
            assert len(self.query_instance) == len(self.hard_answer_set)

        self.length = len(self.query_instance)


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


class MyDataIterator:
    def __init__(self, tasks) -> None:
        self.tasks = tasks


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
