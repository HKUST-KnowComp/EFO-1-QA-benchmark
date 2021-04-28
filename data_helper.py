#!/usr/bin/python3

import os
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fol import parse_foq_formula


class Task:
    def __init__(self, filename):
        self.filename = filename
        self.device = None
        self.query_instance = None
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.i = 0
        self.length = 0
        self._load()
        self.idxlist = np.random.permutation(len(self))

    def to(self, device):
        self.query_instance.to(device)
        self.device = device

    def _load(self):
        dense = self.filename.replace('data', 'tmp').replace('csv', 'pickle')
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
                os.makedirs(os.path.dirname(dense), exist_ok=True)
                print(f"save to {dense}")
                with open(dense, 'wb') as f:
                    pickle.dump(data, f)
            except:
                print(f"can't save to {dense}")

    def __len__(self):
        return self.length

    def setup_iteration(self):
        self.idxlist = np.random.permutation(len(self))

    def batch_estimation_iterator(self, estimator, batch_size):
        assert self.device == estimator.device
        i = 0
        while i < len(self):
            batch_indices = self.idxlist[i: i + batch_size].tolist()
            i += batch_size
            print(i, len(self))
            batch_embedding = self.query_instance.embedding_estimation(
                estimator=estimator,
                batch_indices=batch_indices)
            yield batch_embedding, batch_indices

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

class TaskManager:
    def __init__(self, mode, tasks: List[Task], device):
        self.tasks = tasks
        self.task_iterators = []
        self.mode = mode
        partition = []
        for t in self.tasks:
            t.to(device)
            partition.append(len(t))
        p = np.asarray(partition)
        self.partition = p / p.sum()

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = []
        for i, t in enumerate(self.tasks):
            t.setup_iteration()
            self.task_iterators.append(
                t.batch_estimation_iterator(estimator,
                                            int(batch_size*self.partition[i]))
            )

        while True:
            finish = 0
            data = defaultdict(list)
            for i, it in enumerate(self.task_iterators):
                try:
                    emb, batch_id = next(it)
                    data['emb'].append(emb)
                    if self.mode == 'train':
                        ans_sets = [list(self.tasks[i].answer_set[j]) for j in batch_id]
                        data['answer_set'].extend(ans_sets)
                    else:
                        easy_ans_sets = [list(self.tasks[i].easy_answer_set[j]) for j in batch_id]
                        data['easy_answer_set'].extend(easy_ans_sets)
                        hard_ans_sets = [list(self.tasks[i].hard_answer_set[j]) for j in batch_id]
                        data['hard_answer_set'].extend(hard_ans_sets)

                except StopIteration:
                    finish += 1

            total_emb = torch.cat(data['emb'])
            if self.mode == 'train':
                yield total_emb, data['answer_set']
            else:
                yield total_emb, data['easy_answer_set'], data['hard_answer_set']

            if finish == len(self.tasks): break


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
