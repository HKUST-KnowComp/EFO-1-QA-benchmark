import argparse
import yaml
import os

import torch
import tqdm
from torch import optim

from fol import TransEEstimator, parse_foq_formula, BetaEstimator, BoxEstimator


parser = argparse.ArgumentParser()
parser.add_argument("--cfg")

if __name__ == "__main__":
    mock_dataset = ("[7,8,9]([1,2,2]({1,1,3})&[3,3,4]({6,5,6}))", [[1, 2], [3], [4, 5, 6]])
    X, Y = mock_dataset
    foq_instance = parse_foq_formula(X)
    print(foq_instance.ground_formula)
    model = BetaEstimator(
        nentity=10, nrelation=10, hidden_dim=200, gamma=40, use_cuda=0, dim_list=[100, 100], num_layers=2, negative_size=2)
    '''model = BoxEstimator(nentity=10, nrelation=10,
                         hidden_dim=200, gamma=20, use_cuda=0, entity_dim=100, box_mode=['none', 1])'''
    opt = optim.SGD(model.parameters(), lr=1e-4)

    with tqdm.trange(10000) as t:
        for i in t:
            opt.zero_grad()
            pred = foq_instance.embedding_estimation(estimator=model, batch_indices=[i % 3, (i+1) % 3], device='cpu')
            # pred = foq_instance.embedding_estimation(estimator=model)
            loss = model.criterion(pred, [Y[i % 3], Y[(i+1) % 3]])
            loss.backward()
            opt.step()
            t.set_postfix({'loss': loss.item()})




