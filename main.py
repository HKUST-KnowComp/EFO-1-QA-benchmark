import argparse
import collections
import os
from pprint import pprint

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm.std import trange

from data_helper import TaskManager
from fol import BetaEstimator, BoxEstimator, TransEEstimator, parse_foq_formula
from fol.base import beta_query
from util import (Writer, load_graph, load_task_manager, read_from_yaml,
                  read_indexing, set_global_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/default.yaml', type=str)
parser.add_argument('--prefix', default='dev', type=str)

# from torch.utils.tensorboard import SummaryWriter
# def train_step(model, opt, dataloader, device):
#     iterator = iter(dataloader)
#     # list of tuple, [0] is query, [1] ans, [2] beta_name
#     batch_flattened_query = next(iterator)
#     all_loss = torch.tensor(0, dtype=torch.float)
#     opt.zero_grad()  # TODO: parallelize query
#     # A dict with key of beta_name, value= list of queries
#     query_dict = collections.defaultdict(list)
#     ans_dict = collections.defaultdict(list)
#     for idx in range(len(batch_flattened_query[0])):
#         query, ans, beta_name = batch_flattened_query[0][idx], \
#                                 batch_flattened_query[1][idx], \
#                                     batch_flattened_query[2][idx]
#         query_dict[beta_name].append(query)
#         ans_dict[beta_name].append(ans)
#     for beta_name in query_dict:
#         meta_formula = beta_query[beta_name]
#         query_instance = parse_foq_formula(meta_formula)
#         for query in query_dict[beta_name]:
#             query_instance.additive_ground(query)
#         pred = query_instance.embedding_estimation(estimator=model, device=device)
#         query_loss = model.criterion(pred, ans_dict[beta_name])
#         all_loss += query_loss
#     loss = all_loss.mean()
#     loss.backward()
#     opt.step()
#     log = {
#         'loss': loss.item()
#     }
#     return log

def train_step(model, opt, iterator):
    # list of tuple, [0] is query, [1] ans, [2] beta_name
    opt.zero_grad()  # TODO: parallelize query
    data = next(iterator)
    loss = 0
    for key in data:
        loss += model.criterion(data[key]['emb'], data[key]['answer_set'])
    loss.backward()
    opt.step()
    log = {
        'loss': loss.item()
    }
    return log


def eval_step(model, eval_iterator, device):
    logs = collections.defaultdict(lambda: collections.defaultdict(float))
    with torch.no_grad():
        for data in eval_iterator:
            for key in data:
                pred = data[key]['emb']
                all_entity_loss = model.compute_all_entity_logit(pred)  # batch*nentity
                argsort = torch.argsort(all_entity_loss, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                if device != torch.device('cpu'):
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1).to(
                            device))
                else:
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1))
                # achieve the ranking of all entities
                for i in range(all_entity_loss.shape[0]):
                    easy_ans = data[key]['easy_answer_set'][i]
                    hard_ans = data[key]['hard_answer_set'][i]
                    num_hard = len(hard_ans)
                    num_easy = len(easy_ans)
                    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                    # only take those answers' rank
                    cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if device != torch.device('cpu'):
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).to(device)
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1
                    # filtered setting: +1 for start at 0, -answer_list for ignore other answers

                    cur_ranking = cur_ranking[masks]
                    # only take indices that belong to the hard answers
                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()
                    logs[key]['MRR'] += mrr
                    logs[key]['HITS1'] += h1
                    logs[key]['HITS3'] += h3
                    logs[key]['HITS10'] += h10

                num_query = all_entity_loss.shape[0]
                for metric in logs[key].keys():
                    logs[key][metric] /= num_query
                logs[key]['num_queries'] = num_query

    return logs


# def training(model, opt, train_iterator, valid_iterator, test_iterator, writer, **train_cfg):
#     lr = train_cfg['learning_rate']
#     with tqdm.trange(train_cfg['steps']) as t:
#         for step in t:
#             log = train_step(model, opt, train_iterator, writer)
#             t.set_postfix({'loss': log['loss']})
#             if step % train_cfg['evaluate_every_steps'] and step > 0:
#                 eval_step(model, valid_iterator, 'valid', writer, **train_cfg)
#                 eval_step(model, test_iterator, 'test', writer, **train_cfg)

#             if step >= train_cfg['warm_up_steps']:
#                 lr /= 5
#                 # logging
#                 opt = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, model.parameters()),
#                     lr=lr
#                 )
#                 train_cfg['warm_up_steps'] *= 1.5
#             if step % train_cfg['save_every_steps']:
#                 pass
#             if step % train_cfg['log_every_steps']:
#                 pass

if __name__ == "__main__":

    args = parser.parse_args()
    # parse args and load config
    # configure = read_from_yaml('config/default.yaml')
    configure = read_from_yaml(args.config)
    print("[main] config loaded")
    pprint(configure)

    # initialize my log writer
    case_name = f'{args.prefix}/{ args.config.split("/")[-1].split(".")[0]}'
    # case_name = 'dev/default'
    writer = Writer(case_name=case_name, config=configure, log_path='log')
    # writer = SummaryWriter('./logs-debug/unused-tb')

    # initialize environments
    set_global_seed(configure.get('seed', 0))
    if configure.get('cuda', -1) >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(configure['cuda']))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device('cpu')

    # prepare the procedure configs
    train_config = configure['train']
    train_config['device'] = device
    eval_config = configure['evaluate']
    eval_config['device'] = device

    # load the data
    print("[main] loading the data")
    data_folder = configure['data']['data_folder']
    entity_dict, relation_dict, id2ent, id2rel = read_indexing(data_folder)
    n_entity, n_relation = len(entity_dict), len(relation_dict)
    projection_train, reverse_train = load_graph(
        os.path.join(data_folder, 'train.txt'), entity_dict, relation_dict)

    # get model
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = n_entity, n_relation
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    model_params['device'] = device
    if model_name == 'beta':
        model = BetaEstimator(**model_params)
    elif model_name == 'Box':
        model = BoxEstimator(**model_params)
    model.to(device)

    if 'train' in configure['action']:
        print("[main] load training data")
        tasks = load_task_manager(
            configure['data']['data_folder'], 'train', task_names=train_config['meta_queries'])
        train_tm = TaskManager('train', tasks, device)
        train_iterator = train_tm.build_iterators(model, batch_size=train_config['batch_size'])
    else:
        train_iterator = None

    if 'valid' in configure['action']:
        print("[main] load valid data")
        tasks = load_task_manager(configure['data']['data_folder'], 'valid',
                                  task_names=configure['evaluate']['meta_queries'])
        valid_tm = TaskManager('valid', tasks, device)
        valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
    else:
        valid_iterator = None

    if 'test' in configure['action']:
        print("[main] load test data")
        tasks = load_task_manager(configure['data']['data_folder'], 'test',
                                  task_names=configure['evaluate']['meta_queries'])
        test_tm = TaskManager('test', tasks, device)
        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
    else:
        test_iterator = None

    lr = train_config['learning_rate']
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # exit()

    with trange(1, train_config['steps']+1) as t:
        for step in t:
            # basic training step
            if train_iterator:
                if step >= train_config['warm_up_steps']:
                    lr /= 5
                    # logging
                    opt = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr
                    )
                    train_config['warm_up_steps'] *= 1.5
                try:
                    _log = train_step(model, opt, train_iterator)
                except StopIteration:
                    print("new epoch")
                    train_iterator = train_tm.build_iterators(model, batch_size=train_config['batch_size'])
                    _log = train_step(model, opt, train_iterator)
                t.set_postfix(_log)
                _log['step'] = step
                if step % train_config['log_every_steps'] == 0:
                    writer.append_trace('train', _log)

            if step % train_config['evaluate_every_steps'] == 0 or step == train_config['evaluate_every_steps']:
                # if train_iterator:
                #     _log = eval_step(model, train_iterator)
                #     _log['step'] = step
                #     writer.append_trace('eval_train', _log)

                if valid_iterator:
                    _log = eval_step(model, valid_iterator, device)
                    _log['step'] = step
                    writer.append_trace('eval_valid', _log)

                if test_iterator:
                    _log = eval_step(model, test_iterator, device)
                    _log['step'] = step
                    writer.append_trace('eval_test', _log)

            if step % train_config['evaluate_every_steps'] == 0:
                writer.save_model(model, step)
