import argparse
import collections
import os
from pprint import pprint

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm.std import trange

from data_helper import (SingledirectionalOneShotIterator, TestDataset,
                         TrainDataset)
from fol import BetaEstimator, BoxEstimator, TransEEstimator, parse_foq_formula
from fol.base import beta_query
from util import (Writer, load_graph, load_query, read_from_yaml, read_indexing,
                  set_global_seed)

# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/default.yaml", type=str)
parser.add_argument("--case_pfx", default="debug", type=str)


def train_step(model, opt, dataloader, device):
    iterator = iter(dataloader)
    # list of tuple, [0] is query, [1] ans, [2] beta_name
    batch_flattened_query = next(iterator)
    all_loss = torch.tensor(0, dtype=torch.float)
    opt.zero_grad()  # TODO: parallelize query
    # A dict with key of beta_name, value= list of queries
    query_dict = collections.defaultdict(list)
    ans_dict = collections.defaultdict(list)
    for idx in range(len(batch_flattened_query[0])):
        query, ans, beta_name = batch_flattened_query[0][idx], \
                                batch_flattened_query[1][idx], \
                                    batch_flattened_query[2][idx]
        query_dict[beta_name].append(query)
        ans_dict[beta_name].append(ans)
    for beta_name in query_dict:
        meta_formula = beta_query[beta_name]
        query_instance = parse_foq_formula(meta_formula)
        for query in query_dict[beta_name]:
            query_instance.additive_ground(query)
        pred = query_instance.embedding_estimation(estimator=model, device=device)
        query_loss = model.criterion(pred, ans_dict[beta_name])
        all_loss += query_loss
    loss = all_loss.mean()
    loss.backward()
    opt.step()
    log = {
        'loss': loss.item()
    }
    return log


def eval_step(model, dataloader, device):
    logs = collections.defaultdict(list)
    with torch.no_grad():
        for batch_flattened_query in tqdm.tqdm(dataloader, disable=True):
            # A dict with key of beta_name, value= list of queries
            query_dict = collections.defaultdict(list)
            easy_ans_dict = collections.defaultdict(list)
            hard_ans_dict = collections.defaultdict(list)
            for idx in range(len(batch_flattened_query[0])):
                query, easy_answer, hard_answer, beta_name = batch_flattened_query[0][idx], batch_flattened_query[1][
                    idx], \
                                                             batch_flattened_query[2][idx], batch_flattened_query[3][
                                                                 idx]
                query_dict[beta_name].append(query)
                easy_ans_dict[beta_name].append(easy_answer)
                hard_ans_dict[beta_name].append(hard_answer)
        for beta_name in query_dict:
            meta_formula = beta_query[beta_name]
            query_instance = parse_foq_formula(meta_formula)
            for query in query_dict[beta_name]:
                query_instance.additive_ground(query)
            pred = query_instance.embedding_estimation(estimator=model, device=device)
            all_entity_loss = model.compute_all_entity_logit(
                pred)  # batch*nentity
            argsort = torch.argsort(all_entity_loss, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            #  create a new torch Tensor for batch_entity_range
            if device != torch.device('cpu'):
                ranking = ranking.scatter_(
                    1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).to(
                        device))
            else:
                ranking = ranking.scatter_(
                    1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                   1))
            # achieve the ranking of all entities
            for i in range(all_entity_loss.shape[0]):
                easy_ans = easy_ans_dict[beta_name][i]
                hard_ans = hard_ans_dict[beta_name][i]
                num_hard = len(hard_ans)
                num_easy = len(easy_ans)
                assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                # only take those answers' rank
                cur_ranking = ranking[idx, list(easy_ans) + list(hard_ans)]
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
                logs[beta_name].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'num_hard_answer': num_hard,
                })
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
    # parse args and load config
    args = parser.parse_args()
    configure = read_from_yaml(args.config)
    print("[main] config loaded")
    pprint(configure)

    # initialize my log writer
    case_name = args.case_pfx +'/'+ args.config.split('/')[-1].split('.')[0]
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

    if 'train' in configure['action']:
        print("[main] load training data")
        all_train_data = load_query(
            configure['data']['data_folder'], 'train', train_config['meta_queries'])
        train_dataloader = DataLoader(dataset=TrainDataset(all_train_data),
                                      batch_size=train_config['batch_size'],
                                      num_workers=configure['data']['cpu'],
                                      shuffle=True,
                                      collate_fn=TrainDataset.collate_fn)
    # note shuffle
    else:
        train_dataloader = None

    if 'valid' in configure['action']:
        valid_data = load_query(
            configure['data']['data_folder'], 'valid', eval_config['meta_queries'])
        valid_dataloader = DataLoader(TestDataset(valid_data),
                                      batch_size=eval_config['batch_size'],
                                      num_workers=configure['data']['cpu'],
                                      collate_fn=TestDataset.collate_fn)
    else:
        valid_dataloader = None

    if 'test' in configure['action']:
        test_data = load_query(
            configure['data']['data_folder'], 'test', eval_config['meta_queries'])
        test_dataloader = DataLoader(TestDataset(test_data),
                                     batch_size=eval_config['batch_size'],
                                     num_workers=configure['data']['cpu'],
                                     collate_fn=TestDataset.collate_fn)
    else:
        test_dataloader = None
    
    exit()

    # get model
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = n_entity, n_relation
    model_params['device'] = device
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    if model_name == 'beta':
        model = BetaEstimator(**model_params)
    elif model_name == 'Box':
        model = BoxEstimator(**model_params)
    model.to(device)

    # optimizer = torch.optim.Adam(
    # filter(lambda p: p.requires_grad, model.parameters()),
    # lr=configure['train']['learning_rate']
    # )

    # training(model, optimizer, train_iterator=train_dataloader, valid_iterator=valid_dataloader,
    #  test_iterator=test_dataloader, writer=writer, **train_config)
    # the main iteration
    lr = train_config['learning_rate']
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    with trange(1, train_config['steps']+1) as t:
        for step in t:
            # basic training step
            if train_dataloader:
                if step >= train_config['warm_up_steps']:
                    lr /= 5
                    # logging
                    opt = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr
                    )
                    train_config['warm_up_steps'] *= 1.5
                _log = train_step(model, opt, train_dataloader, device)
                _log['step'] = step
                if step % train_config['log_every_steps'] == 0:
                    writer.append_trace('train', _log)

            if step % train_config['evaluate_every_steps'] == 0 or step == train_config['evaluate_every_steps']:
                if train_dataloader:
                    _log = eval_step(model, train_dataloader, device)
                    _log['step'] = step
                    writer.append_trace('eval_train', _log)

                if valid_dataloader:
                    _log = eval_step(model, valid_dataloader, device)
                    _log['step'] = step
                    writer.append_trace('eval_valid', _log)

                if test_dataloader:
                    _log = eval_step(model, test_dataloader, device)
                    _log['step'] = step
                    writer.append_trace('eval_test', _log)

            if step % train_config['evaluate_every_steps'] == 0:
                writer.save_model(model, step)

