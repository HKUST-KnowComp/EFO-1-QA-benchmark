import argparse
import yaml
import os

import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fol import TransEEstimator, parse_foq_formula, BetaEstimator
from utils.util import *

parser = argparse.ArgumentParser()
parser.add_argument("--cfg")
writer = SummaryWriter('./logs-debug/unused-tb')


def training(model, opt, dataloader, writer, **train_cfg):
    lr = train_cfg['learning_rate']
    for step in range(train_cfg['steps']):
        train_step(model, opt, dataloader, writer)
        if step % train_cfg['evaluate_every_steps'] and step > 0 :
            test_step(model, data, writer)
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


def train_step(model, opt, data, writer):
    pass


def test_step(model, data, writer):
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
    # Train_dataloader = DataLoader()
    training(model, optimizer, writer, **configure['train'])

    data = load_our_query(os.path.join(data_folder, 'train_1p.csv'))
    # A dict of dict, with keys={queries, id, answer_set}, the second keys are ids
    mock_dataset = ("[7,8,9]([1,2,2]({1,1,3})&[3,3,4]({6,5,6}))", [[1, 2], [3], [4, 5, 6]])
    X, Y = mock_dataset
    foq_instance = parse_foq_formula(X)
    print(foq_instance.ground_formula)

    with tqdm.trange(10000) as t:
        for i in t:
            optimizer.zero_grad()
            pred = foq_instance.embedding_estimation(estimator=model, batch_indices=[i % 3], device='cpu')
            # pred = foq_instance.embedding_estimation(estimator=model)
            loss = model.criterion(pred, Y)
            loss.backward()
            optimizer.step()
            t.set_postfix({'loss': loss.item()})




