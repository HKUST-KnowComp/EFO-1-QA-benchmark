import argparse
import yaml
import os

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from fol import TransEEstimator, parse_foq_formula, BetaEstimator
from utils.util import *
from utils.dataloader import TestDataset,TrainDataset, SingledirectionalOneShotIterator

parser = argparse.ArgumentParser()
parser.add_argument("--cfg")
writer = SummaryWriter('./logs-debug/unused-tb')


def training(model, opt, train_iterator, test_iterator, writer, **train_cfg):
    lr = train_cfg['learning_rate']
    with tqdm.trange(train_cfg['steps']) as t:
        for step in t:
            log = train_step(model, opt, train_iterator, writer)
            t.set_postfix({'loss': log['loss']})
            if step % train_cfg['evaluate_every_steps'] and step > 0:
                test_step(model, test_iterator, writer)
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
    for idx in range(len(batch_flattened_query[0])):
        query = batch_flattened_query[0][idx]
        ans = batch_flattened_query[1][idx]
        query_instance = parse_foq_formula(query)
        pred = query_instance.embedding_estimation(estimator=model)
        query_loss = model.criterion(pred, [ans])
        all_loss += query_loss
    loss = all_loss.mean()
    loss.backward()
    opt.step()
    log = {
        'loss': loss.item()
    }
    return log


def test_step(model, iterator, writer):
    pass


if __name__ == "__main__":
    yaml_path = 'config/default.yaml'
    configure = read_from_yaml(yaml_path)
    set_global_seed(configure['seed'])
    start_time = parse_time()
    data_folder = configure['data']['data_folder']

    entity_dict, relation_dict, id2ent, id2rel = read_indexing(data_folder)
    nentity, nrelation = len(entity_dict), len(relation_dict)
    projection_train, reverse_train = load_graph(os.path.join(data_folder, 'train.txt'), entity_dict, relation_dict)

    model_name = configure['estimator']['embedding']
    hyperparameters = configure['estimator'][model_name]
    hyperparameters['nentity'], hyperparameters['nrelation'] = nentity, nrelation
    hyperparameters['use_cuda'] = configure['cuda']
    if model_name == 'beta':
        model = BetaEstimator(**hyperparameters)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configure['train']['learning_rate']
    )

    all_data = load_our_query(configure['data']['data_folder'], 'train', configure['train']['meta_queries'])
    Train_Dataloader = SingledirectionalOneShotIterator(
        DataLoader(dataset=TrainDataset(all_data), batch_size=configure['train']['batch_size'],
                   num_workers=configure['data']['cpu'], shuffle=True, collate_fn=TrainDataset.collate_fn))
    # note shuffle

    training(model, optimizer, train_iterator=Train_Dataloader, test_iterator=Train_Dataloader,
             writer=writer, **configure['train'])

    mock_dataset = ("[7,8,9]([1,2,2]({1,1,3})&[3,3,4]({6,5,6}))", [[1, 2], [3], [4, 5, 6]])
    X, Y = mock_dataset
    foq_instance = parse_foq_formula(X)





