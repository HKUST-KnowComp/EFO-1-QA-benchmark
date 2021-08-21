#!/usr/bin/python3

import os
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from tqdm import tqdm

from fol import parse_foq_formula, parse_formula, beta_query_v2
all_normal_form = ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'diff', 'DNF+diff', 'DNF+MultiIU', 'DNF+MultiIUD']


class Task:
    def __init__(self, filename, task_betaname):
        self.filename = filename
        self.device = None
        self.query_instance = None
        self.beta_name = task_betaname
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.i = 0
        self.length = 0
        self._load()
        self.idxlist = np.random.permutation(len(self))
        # self.idxlist = np.arange(len(self))

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
            self.query_instance = parse_formula(beta_query_v2[self.beta_name])
            self._parse(df)
            data = {'query_instance': self.query_instance, 'answer_set': self.answer_set,
                    'easy_answer_set': self.easy_answer_set, 'hard_answer_set': self.hard_answer_set}
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
        # self.idxlist = np.arange(len(self))

    def batch_estimation_iterator(self, estimator, batch_size):
        assert self.device == estimator.device
        i = 0
        while i < len(self):
            batch_indices = self.idxlist[i: i + batch_size].tolist()
            i += batch_size
            batch_embedding = self.query_instance.embedding_estimation(
                estimator=estimator,
                batch_indices=batch_indices)
            yield batch_embedding, batch_indices

    def _parse(self, df):
        for q in tqdm(df['query']):
            self.query_instance.additive_ground(json.loads(q))

        if 'answer_set' in df.columns:
            self.answer_set = df.answer_set.map(lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.answer_set)

        if 'easy_answer_set' in df.columns:
            self.easy_answer_set = df.easy_answer_set.map(
                lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.easy_answer_set)

        if 'hard_answer_set' in df.columns:
            self.hard_answer_set = df.hard_answer_set.map(
                lambda x: list(eval(x))).tolist()
            assert len(self.query_instance) == len(self.hard_answer_set)

        self.length = len(self.query_instance)


class TaskManager:
    def __init__(self, mode, tasks: List[Task], device):
        self.tasks = {t.query_instance.formula: t for t in tasks}
        self.task_iterators = {}
        self.mode = mode
        partition = []
        for t in self.tasks:
            self.tasks[t].to(device)
            partition.append(len(self.tasks[t]))
        p = np.asarray(partition)
        self.partition = p / p.sum()

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = {}
        for i, tmf in enumerate(self.tasks):
            self.tasks[tmf].setup_iteration()
            self.task_iterators[tmf] = \
                self.tasks[tmf].batch_estimation_iterator(
                    estimator,
                    int(batch_size * self.partition[i]))

        while True:
            finish = 0
            data = defaultdict(dict)
            for tmf in self.task_iterators:
                try:
                    emb, batch_id = next(self.task_iterators[tmf])
                    data[tmf]['emb'] = emb
                    if self.mode == 'train':
                        ans_sets = [self.tasks[tmf].answer_set[j] for j in batch_id]
                        data[tmf]['answer_set'] = ans_sets
                    else:
                        easy_ans_sets = [self.tasks[tmf].easy_answer_set[j] for j in batch_id]
                        data[tmf]['easy_answer_set'] = easy_ans_sets
                        hard_ans_sets = [self.tasks[tmf].hard_answer_set[j] for j in batch_id]
                        data[tmf]['hard_answer_set'] = hard_ans_sets

                except StopIteration:
                    finish += 1

            if finish == len(self.tasks):
                break

            yield data


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


class BenchmarkTaskManager:
    def __init__(self,  data_folder: str, task_id: int, device, model):
        all_formula = pd.read_csv('data/generated_formula_anchor_node=3.csv')
        self.task_id = task_id
        self.tasks, self.form2formula = {}, {}
        self.all_formula, self.allowed_formula = set(), set()
        id_str = str(task_id)
        self.id_str = '0' * (4 - len(id_str)) + id_str
        filename = os.path.join(data_folder, f'data-type{self.id_str}.csv')
        real_index = all_formula.loc[all_formula['formula_id'] == f'type{self.id_str}'].index[0]  # index != formula id
        for normal_form in all_normal_form:
            formula = all_formula[normal_form][real_index]
            self.form2formula[normal_form] = formula
            self.all_formula.add(formula)
        print(f'[data] load query from file {filename}')
        self._load(filename, model)
        self.task_iterators = {}
        for t in self.tasks:
            self.tasks[t].set_up(device, self.len)
        self.partition = [1 / len(self.tasks) for i in range(len(self.tasks))]

    def _load(self, filename, model):
        dense = filename.replace('data', 'tmp').replace('csv', 'pickle')
        if os.path.exists(dense):
            print("load from existed files")
            with open(dense, 'rb') as f:
                data = pickle.load(f)
                self.easy_answer_set = data['easy_answer_set']
                self.hard_answer_set = data['hard_answer_set']
                self.len = len(self.easy_answer_set)
                for formula in self.all_formula:
                    query_instance = data[formula]
                    try:
                        query_instance.to(model.device)
                        pred_emb = query_instance.embedding_estimation(estimator=model, batch_indices=[0, 1])
                        assert pred_emb.ndim == 2 + ('u' in formula or 'U' in formula)
                        self.allowed_formula.add(formula)
                    except (AssertionError, RuntimeError):
                        pass
                    if formula in self.allowed_formula:
                        self.tasks[formula] = BenchmarkTask(data[formula])
                    assert len(data[formula]) == self.len
        else:
            df = pd.read_csv(filename)
            self.len = len(df)
            loaded = {formula: False for formula in self.all_formula}
            if 'easy_answers' in df.columns:
                self.easy_answer_set = df.easy_answers.map(
                    lambda x: list(eval(x))).tolist()
                assert self.len == len(self.easy_answer_set)
            if 'hard_answers' in df.columns:
                self.hard_answer_set = df.hard_answers.map(
                    lambda x: list(eval(x))).tolist()
                assert self.len == len(self.hard_answer_set)
            data = {'easy_answer_set': self.easy_answer_set, 'hard_answer_set': self.hard_answer_set}
            for normal_form in all_normal_form:
                formula = self.form2formula[normal_form]
                if not loaded[formula]:
                    query_instance = parse_formula(formula)
                    for q in df[normal_form]:
                        query_instance.additive_ground(json.loads(q))
                    data[formula] = query_instance
                    query_instance.to(model.device)
                    try:
                        pred_emb = query_instance.embedding_estimation(estimator=model, batch_indices=[0, 1])
                        assert pred_emb.ndim == 2 + ('u' in formula or 'U' in formula)
                        self.allowed_formula.add(formula)
                    except (AssertionError, RuntimeError):
                        pass
                    if formula in self.allowed_formula:
                        self.tasks[formula] = BenchmarkTask(query_instance)
                    loaded[formula] = True
            try:
                os.makedirs(os.path.dirname(dense), exist_ok=True)
                print(f"save to {dense}")
                with open(dense, 'wb') as f:
                    pickle.dump(data, f)
            except:
                print(f"can't save to {dense}")

    def build_iterators(self, estimator, batch_size):
        self.task_iterators = {}
        for i, tmf in enumerate(self.tasks):
            self.task_iterators[tmf] = \
                self.tasks[tmf].batch_estimation_iterator(
                    estimator,
                    int(batch_size * self.partition[i]))

        while True:
            finish = 0
            data = defaultdict(dict)
            for tmf in self.task_iterators:
                try:
                    emb, batch_id = next(self.task_iterators[tmf])
                    data[tmf]['emb'] = emb
                    easy_ans_sets = [self.easy_answer_set[j] for j in batch_id]
                    data[tmf]['easy_answer_set'] = easy_ans_sets
                    hard_ans_sets = [self.hard_answer_set[j] for j in batch_id]
                    data[tmf]['hard_answer_set'] = hard_ans_sets

                except StopIteration:
                    finish += 1

            if finish == len(self.tasks):
                break

            yield data


class BenchmarkTask:
    def __init__(self, query_instance):
        self.query_instance = query_instance
        self.device = None
        self.answer_set = None
        self.easy_answer_set = None
        self.hard_answer_set = None
        self.i = 0
        self.length = 0
        self.idxlist = np.arange(len(self))

    def set_up(self, device, length):
        self.length = length
        self.query_instance.to(device)
        self.device = device
        self.idxlist = np.arange(len(self))

    def __len__(self):
        return self.length

    def batch_estimation_iterator(self, estimator, batch_size):
        assert self.device == estimator.device
        i = 0
        while i < len(self):
            batch_indices = self.idxlist[i: i + batch_size].tolist()
            i += batch_size
            batch_embedding = self.query_instance.embedding_estimation(
                estimator=estimator,
                batch_indices=batch_indices)
            yield batch_embedding, batch_indices

