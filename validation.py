"""
This file validate the correctness of our framework by
comparing the BetaE model during optimization procedure.
"""


import argparse
import logging
import pickle
from os.path import join
from typing import Dict, List, Set, Tuple
import json

import yaml
from torch.utils.data.dataloader import DataLoader
import torch
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence

from fol.appfoq import BetaEstimator4V
from fol.foq_v2 import parse_formula
from fol import beta_query_v2
from main import compute_final_loss
from transform_beta_data import transform_json_query
from KGReasoning.dataloader import (SingledirectionalOneShotIterator,
                                    TrainDataset)
from KGReasoning.models import KGReasoning
from KGReasoning.util import flatten_query

parser = argparse.ArgumentParser()

parser.add_argument("--save_intermediate_results",
                    action="store_true",
                    default=False)
parser.add_argument("--begin_checkpoint",
                    type=str,
                    default="")
parser.add_argument("--model_config_file",
                    type=str,
                    default="config/validation.yaml")
parser.add_argument("--cuda", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dataset_folder",
                    type=str,
                    default='data/FB15k-237-betae')


query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }


def ref_train_step(beta_train_queries: List[Tuple[str, Tuple]],
                   train_answers: Dict[Tuple, Set],
                   model: KGReasoning,
                   optimizer,
                   nentity,
                   nrelation,
                   negative_sample_size) -> Dict:
    """
    Conduct a train step from KG reasoning
    """
    train_iterator = SingledirectionalOneShotIterator(DataLoader(
        TrainDataset(beta_train_queries,
                     nentity,
                     nrelation,
                     negative_sample_size,
                     train_answers),
        batch_size=len(beta_train_queries),
        shuffle=False,
        num_workers=1,
        collate_fn=TrainDataset.collate_fn
    ))
    log = model.train_step(model, optimizer, train_iterator, args, step=None)
    return log


def our_train_step(batch_train_queries,
                   positive_sample,
                   negative_sample,
                   subsampling_weight,
                   model: BetaEstimator4V,
                   optimizer) -> Dict:
    optimizer.zero_grad()
    positive_logits, negative_logits = [], []
    for query, query_structure in batch_train_queries:
        query_name = query_name_dict[query_structure]
        formula = beta_query_v2[query_name]
        query_instance = parse_formula(formula)
        query_instance.additive_ground(
            json.loads(
                transform_json_query(query, query_name)))
        pred_embedding = query_instance.embedding_estimation(model)
        pred_alpha, pred_beta = torch.chunk(pred_embedding, 2, dim=-1)
        pred_dist = Beta(pred_alpha, pred_beta)
        pred_dist_unsqueezed = Beta(pred_alpha.unsqueeze(1),
                                    pred_beta.unsqueeze(1))

        positive_embedding = model.get_entity_embedding(
            positive_sample.view(-1)).view(-1, 2 * model.entity_dim)
        positive_logit = model.compute_logit(positive_embedding, pred_dist)
        positive_logits.append(positive_logit)

        negative_embedding = model.get_entity_embedding(
            negative_sample.view(-1)).view(
                -1, model.negative_size, 2 * model.entity_dim)
        negative_logit = model.compute_logit(negative_embedding, 
                                             pred_dist_unsqueezed)
        negative_logits.append(negative_logit)
    positive_logits = torch.cat(positive_logits) 
    negative_logits = torch.cat(negative_logits)
    pos_loss, neg_loss = compute_final_loss(positive_logits, negative_logits, 
                                            subsampling_weight)
    loss = pos_loss + neg_loss
    loss /= 2
    loss.backward()
    optimizer.step()
    log = {
        'positive_sample_loss': pos_loss.item(),
        'negative_sample_loss': neg_loss.item(),
        'loss': loss.item(),
    }
    return log
    
def recursive_getattr(obj, k):
    attr_list = k.split('.')
    _obj = obj
    for attr in attr_list:
        _obj = getattr(_obj, attr)
    return _obj


def check_model_sync(ref_model, our_model):
    ref_state_dict = ref_model.state_dict()
    our_state_dict = our_model.state_dict()
    tensor_diff_keys = []
    grad_diff_keys = []
    for k in ref_state_dict:
        tensor_ref = ref_state_dict[k]
        grad_ref = recursive_getattr(ref_model, k).grad
        tensor_our = our_state_dict[k]
        grad_our = recursive_getattr(our_model, k).grad
        if not (tensor_our == tensor_ref).all():
            tensor_diff_keys.append(k)
        
        if grad_our is not None and not (grad_our == grad_ref).all():
            grad_diff_keys.append(k)
    return tensor_diff_keys, grad_diff_keys 
        
    

# def ref_eval_step(batch_data, model) -> Dict:
    # pass


# def out_eval_step(batch_data, model) -> Dict:
    # pass


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(filename="logs/kgr_validation.log",
                        filemode="wt",
                        level=logging.INFO)
    with open(args.model_config_file, 'rt') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    ref_model_config = model_config['ref']
    our_model_config = model_config['our']
    ref_model = KGReasoning(query_name_dict=query_name_dict,
                            **ref_model_config)
    our_model = BetaEstimator4V(**our_model_config)
    # model syncronization
    our_model.load_state_dict(ref_model.state_dict())
    check_model_sync(ref_model, our_model)
    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=args.lr)
    our_opt = torch.optim.Adam(our_model.parameters(), lr=args.lr)
    with open(f"{args.dataset_folder}/stats.txt", 'rt') as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    train_queries = pickle.load(
        open(join(args.dataset_folder, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(join(args.dataset_folder, "train-answers.pkl"), 'rb'))
    # valid_queries = pickle.load(open(join(args.dataset_folder, "valid-queries.pkl"), 'rb'))
    # valid_hard_answers = pickle.load(open(join(args.dataset_folder, "valid-hard-answers.pkl"), 'rb'))
    # valid_easy_answers = pickle.load(open(join(args.dataset_folder, "valid-easy-answers.pkl"), 'rb'))
    # test_queries = pickle.load(open(join(args.dataset_folder, "test-queries.pkl"), 'rb'))
    # test_hard_answers = pickle.load(open(join(args.dataset_folder, "test-hard-answers.pkl"), 'rb'))
    # test_easy_answers = pickle.load(open(join(args.dataset_folder, "test-easy-answers.pkl"), 'rb'))
    flat_train_queries = flatten_query(train_queries)
    train_size = len(flat_train_queries)


    for i in range(0, train_size, args.batch_size):
        batch_train_queries = flat_train_queries[i: i+args.batch_size]
        ref_log = ref_train_step(batch_train_queries,
                                 train_answers,
                                 ref_model,
                                 ref_opt,
                                 nentity,
                                 nrelation,
                                 negative_sample_size=128)

        positive_sample = ref_log['positive_sample']
        negative_sample = ref_log['negative_sample']
        subsampling_weight = ref_log['subsampling_weight']
        
        our_log = our_train_step(batch_train_queries,
                                 positive_sample,
                                 negative_sample,
                                 subsampling_weight,
                                 our_model,
                                 our_opt)
        
        for k in our_log:
            assert our_log[k] == ref_log[k]
            logging.info(f"step {i}, {k} = {our_log[k]} is validated for ref and our model")
        
        tdk, gdk = check_model_sync(ref_model, our_model)
        
        if len(tdk) != 0 or len(gdk) != 0:
            for k in tdk:
                logging.error(f"step {i} tensor value {k} differs")
            for k in gdk:
                logging.error(f"step {i} gradient value {k} differs")
        else:
            logging.info(f"step {i}, tensors and gradients of two models are the same")
            
        