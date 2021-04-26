import argparse
import yaml
import os
import collections

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from fol import TransEEstimator, parse_foq_formula, BetaEstimator, BoxEstimator
from utils.util import *
from utils.dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from fol.base import beta_query


parser = argparse.ArgumentParser()
parser.add_argument("--cfg")
writer = SummaryWriter('./logs-debug/unused-tb')


def training(model, opt, train_iterator, valid_iterator, test_iterator, writer, **train_cfg):
    lr = train_cfg['learning_rate']
    with tqdm.trange(train_cfg['steps']) as t:
        for step in t:
            log = train_step(model, opt, train_iterator, writer)
            t.set_postfix({'loss': log['loss']})
            if step % train_cfg['evaluate_every_steps'] and step > 0:
                test_step(model, valid_iterator, 'valid', writer, **train_cfg)
                test_step(model, test_iterator, 'test', writer, **train_cfg)

            if step >= train_cfg['warm_up_steps']:
                lr /= 5
                # logging
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr
                )
                train_cfg['warm_up_steps'] *= 1.5
            if step % train_cfg['save_every_steps']:
                pass
            if step % train_cfg['log_every_steps']:
                pass


def train_step(model, opt, iterator, writer):
    batch_flattened_query = next(iterator)  # list of tuple, [0] is query, [1] ans, [2] beta_name
    all_loss = torch.tensor(0, dtype=torch.float)
    opt.zero_grad()  # TODO: parallelize query
    query_dict = collections.defaultdict(list)  # A dict with key of beta_name, value= list of queries
    ans_dict = collections.defaultdict(list)
    for idx in range(len(batch_flattened_query[0])):
        query, ans, beta_name = batch_flattened_query[0][idx],\
                                batch_flattened_query[1][idx], batch_flattened_query[2][idx]
        query_dict[beta_name].append(query)
        ans_dict[beta_name].append(ans)
    for beta_name in query_dict:
        meta_formula = beta_query[beta_name]
        query_instance = parse_foq_formula(meta_formula)
        for query in query_dict[beta_name]:
            query_instance.additive_ground(query)
        pred = query_instance.embedding_estimation(estimator=model)
        query_loss = model.criterion(pred, ans_dict[beta_name])
        all_loss += query_loss
    loss = all_loss.mean()
    loss.backward()
    opt.step()
    log = {
        'loss': loss.item()
    }
    return log


def test_step(model, iterator, mode, writer, **cfg):
    logs = collections.defaultdict(list)
    with torch.no_grad():
        for batch_flattened_query in tqdm.tqdm(iterator, disable=True):
            query_dict = collections.defaultdict(list)  # A dict with key of beta_name, value= list of queries
            easy_ans_dict = collections.defaultdict(list)
            hard_ans_dict = collections.defaultdict(list)
            for idx in range(len(batch_flattened_query[0])):
                query, easy_answer, hard_answer, beta_name = batch_flattened_query[0][idx], batch_flattened_query[1][idx], \
                                                       batch_flattened_query[2][idx], batch_flattened_query[3][idx]
                query_dict[beta_name].append(query)
                easy_ans_dict[beta_name].append(easy_answer)
                hard_ans_dict[beta_name].append(hard_answer)
        for beta_name in query_dict:
            meta_formula = beta_query[beta_name]
            query_instance = parse_foq_formula(meta_formula)
            for query in query_dict[beta_name]:
                query_instance.additive_ground(query)
            pred = query_instance.embedding_estimation(estimator=model)
            all_entity_loss = model.compute_all_entity_logit(pred)  # batch*nentity
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
                cur_ranking = ranking[idx, list(easy_ans) + list(hard_ans)]  # only take those answers' rank
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


if __name__ == "__main__":
    yaml_path = 'config/default.yaml'
    configure = read_from_yaml(yaml_path)
    set_global_seed(configure['seed'])
    start_time = parse_time()
    data_folder = configure['data']['data_folder']

    entity_dict, relation_dict, id2ent, id2rel = read_indexing(data_folder)
    nentity, nrelation = len(entity_dict), len(relation_dict)
    projection_train, reverse_train = load_graph(os.path.join(data_folder, 'train.txt'), entity_dict, relation_dict)

    if configure['cuda'] >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(configure['cuda']))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device('cpu')

    model_name = configure['estimator']['embedding']
    hyperparameters = configure['estimator'][model_name]
    hyperparameters['nentity'], hyperparameters['nrelation'] = nentity, nrelation
    hyperparameters['device'] = device
    hyperparameters['negative_sample_size'] = configure['train']['negative_sample_size']
    if model_name == 'beta':
        model = BetaEstimator(**hyperparameters)
    elif model_name == 'Box':
        model = BoxEstimator(**hyperparameters)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configure['train']['learning_rate']
    )

    all_train_data = load_our_query(configure['data']['data_folder'], 'train', configure['train']['meta_queries'])
    Train_Dataloader = SingledirectionalOneShotIterator(
        DataLoader(dataset=TrainDataset(all_train_data), batch_size=configure['train']['batch_size'],
                   num_workers=configure['data']['cpu'], shuffle=True, collate_fn=TrainDataset.collate_fn))
    # note shuffle

    valid_data = load_our_query(configure['data']['data_folder'], 'valid', configure['evaluate']['tasks'])
    Valid_Dataloader = DataLoader(TestDataset(valid_data), batch_size=configure['evaluate']['batch_size'],
                                  num_workers=configure['data']['cpu'], collate_fn=TestDataset.collate_fn)

    test_data = load_our_query(configure['data']['data_folder'], 'test', configure['evaluate']['tasks'])
    Test_Dataloader = DataLoader(TestDataset(test_data), batch_size=configure['evaluate']['batch_size'],
                                 num_workers=configure['data']['cpu'], collate_fn=TestDataset.collate_fn)

    train_hyper = configure['train']
    train_hyper['device'] = device

    training(model, optimizer, train_iterator=Train_Dataloader, valid_iterator=Valid_Dataloader,
             test_iterator=Test_Dataloader, writer=writer, **train_hyper)








