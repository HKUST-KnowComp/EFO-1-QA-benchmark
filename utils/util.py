import hashlib
import json
import os
import pickle
import random
import time
from os.path import join
from shutil import rmtree
import copy
import collections
import pandas as pd

import numpy as np
from pandas.core.frame import DataFrame
import torch
import yaml

from data_helper import Task
from fol.sampler import read_indexing, load_data

dir_path = os.path.dirname(os.path.realpath(__file__))


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def flatten(l): return sum(map(flatten, l), []
                           ) if isinstance(l, tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


class Writer:

    _log_path = join(dir_path, 'log')

    def __init__(self, case_name, config, log_path=None, postfix=True, tb_writer=None):
        if isinstance(config, dict):
            self.meta = config
        else:
            self.meta = vars(config)
        self.time = time.time()
        self.meta['time'] = self.time
        self.idstr = case_name
        self.column_name = {}
        if postfix:
            self.idstr += time.strftime("%y%m%d.%H:%M:%S", time.localtime()) + \
                    hashlib.sha1(str(self.meta).encode('UTF-8')).hexdigest()[:8]

        self.log_path = log_path if log_path else self._log_path
        if os.path.exists(self.case_dir):
            rmtree(self.case_dir)
        os.makedirs(self.case_dir, exist_ok=False)

        with open(self.metaf, 'wt') as f:
            json.dump(self.meta, f)

    def append_trace(self, trace_name, data):
        if trace_name not in self.column_name:
            self.column_name[trace_name] = list(data.keys())
            assert len(self.column_name[trace_name]) > 0
        if not os.path.exists(self.tracef(trace_name)):
            with open(self.tracef(trace_name), 'at') as f:
                f.write(','.join(self.column_name[trace_name]) + '\n')
        with open(self.tracef(trace_name), 'at') as f:
            f.write(','.join([str(data[c]) for c in self.column_name[trace_name]]) + '\n')

    def save_pickle(self, obj, name):
        with open(join(self.case_dir, name), 'wb') as f:
            pickle.dump(obj, f)

    def save_array(self, arr, name):
        np.save(join(self.case_dir, name), arr)

    def save_json(self, obj, name):
        if not name.endswith('json'):
            name += '.json'
        with open(join(self.case_dir, name), 'wt') as f:
            json.dump(obj, f)

    def save_model(self, model: torch.nn.Module, opt, step, warm_up_step, lr):
        print("saving model : ", step)
        device = model.device
        save_data = {'model_parameter': model.cpu().state_dict(), 'optimizer_parameter': opt.state_dict(),
                     'warm_up_steps': warm_up_step, 'learning_rate': lr}
        torch.save(save_data, self.modelf(step))
        model.to(device)

    def save_plot(self, fig, name):
        fig.savefig(join(self.case_dir, name))

    @property
    def case_dir(self):
        return join(self.log_path, self.idstr)

    @property
    def metaf(self):
        return join(self.case_dir, 'meta.json')

    def tracef(self, name):
        return join(self.case_dir, '{}.csv'.format(name))

    def modelf(self, e):
        return join(self.case_dir, '{}.ckpt'.format(e))


def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd, Loader=yaml.FullLoader)
    return data

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def read_indexing(data_path):
    ent2id = pickle.load(open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(data_path, "rel2id.pkl"), 'rb'))
    id2ent = pickle.load(open(os.path.join(data_path, "id2ent.pkl"), 'rb'))
    id2rel = pickle.load(open(os.path.join(data_path, "id2rel.pkl"), 'rb'))
    return ent2id, rel2id, id2ent, id2rel


def load_task_manager(data_folder, mode, task_names=[]):
    all_data = []
    tasks = []
    if task_names:
        for task_name in task_names:
            filename = os.path.join(data_folder, f'{mode}_{task_name}.csv')
            print(f'[data] load query from file {filename}')
            task = Task(filename, task_name)
            tasks.append(task)
    return tasks


def parse_ans_set(answer_set: str):
    ans_list = answer_set.strip().split(',')
    if len(ans_list) > 1:
        for i in range(len(ans_list)):
            if i == 0:
                ans_list[i] = int(ans_list[i][1:])
            elif i == len(ans_list) - 1:
                ans_list[i] = int(ans_list[i][:-1])
            else:
                ans_list[i] = int(ans_list[i])
    elif len(ans_list) == 1 and ans_list[0] != 'set()':
        ans_list[0] = int(ans_list[0][1:-1])
    else:
        assert ans_list[0] == 'set()'
        ans_list = []

    return ans_list


def load_rawdata_with_indexing(pickle_datapath, rawdata_path):
    entity_dict, relation_dict, id2ent, id2rel = read_indexing(pickle_datapath)
    proj_none = collections.defaultdict(lambda: collections.defaultdict(set))
    reverse_none = collections.defaultdict(lambda: collections.defaultdict(set))
    proj_train, reverse_train = load_data(os.path.join(rawdata_path, "train.txt"),
                                          entity_dict, relation_dict, proj_none, reverse_none)
    proj_valid, reverse_valid = load_data(os.path.join(rawdata_path, "valid.txt"),
                                          entity_dict, relation_dict, proj_train, reverse_train)
    proj_test, reverse_test = load_data(os.path.join(rawdata_path, "test.txt"),
                                        entity_dict, relation_dict, proj_valid, reverse_valid)
    return entity_dict, relation_dict, proj_train, reverse_train, proj_valid, reverse_valid, proj_test, reverse_test


def read_indexed_graph(input_edge_file, projection_origin=None, reverse_projection_origin=None):
    if projection_origin is None:
        projection_origin = collections.defaultdict(lambda: collections.defaultdict(set))
    if reverse_projection_origin is None:
        reverse_projection_origin = collections.defaultdict(lambda: collections.defaultdict(set))
    projections = copy.deepcopy(projection_origin)
    reverse = copy.deepcopy(reverse_projection_origin)
    with open(input_edge_file, 'r', errors='ignore') as infile:
        for line in infile.readlines():
            e1, r, e2 = line.strip().split('\t')
            projections[int(e1)][int(r)].add(int(e2))
            reverse[int(e2)][int(r)].add(int(e1))
    return projections, reverse


def load_indexed_graph(input_folder):
    projs_train, rprojs_train = read_indexed_graph(os.path.join(input_folder, 'train.txt'))
    projs_valid, rprojs_valid = read_indexed_graph(os.path.join(input_folder, 'valid.txt'), projs_train, rprojs_train)
    projs_test, rprojs_test = read_indexed_graph(os.path.join(input_folder, 'test.txt'), projs_valid, rprojs_valid)
    return projs_train, rprojs_train, projs_valid, rprojs_valid, projs_test, rprojs_test
