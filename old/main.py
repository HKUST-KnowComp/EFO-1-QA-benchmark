#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import (SingledirectionalOneShotIterator, TestDataset,
                        TrainDataset)
from models import KGReasoning
from util import (eval_tuple, flatten_query, list2tuple, parse_time,
                  set_global_seed)

"""

query = (q, q)
q = e|q|('r'|'n')

"""

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
name_query_dict = {value: key for key, value in query_name_dict.items()}
# ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
all_tasks = list(name_query_dict.keys())


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', default=0, type=int, help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_test_in_train',
                        action='store_true', help="test_overfitting")

    parser.add_argument('--data_path', type=str,
                        default="../data/FB15k-237-betae", help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128,
                        type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400,
                        type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=60,
                        type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512,
                        type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1,
                        type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10,
                        type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000,
                        type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int,
                        help="no need to set manually, will configure automatically")

    parser.add_argument('--save_checkpoint_steps', default=50000,
                        type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=30000, type=int,
                        help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int,
                        help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000,
                        type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0,
                        help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0,
                        help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='beta', type=str, choices=[
                        'vec', 'box', 'beta', 'discrete'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--number_of_entities', default=-1,
                        type=int, help="how many entities we train")
    parser.add_argument('--number_of_queries', default=-1,
                        type=int, help="how many queries we use")
    parser.add_argument('--test_tasks', default='3p.3i',
                        type=str, help="tasks to be tested")
    parser.add_argument('--train_tasks', default='1p.2i', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str,
                        help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-GMM', '--GMM_mode', default="(0,5)",
                        type=str, help='(use_Gaussian,topk) for GMM')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str,
                        help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('-discretem', '--discrete_mode', default="(4,6,20,2)",
                        type=str, help="(relation_dim,hidden_dim,num_layer)")
    parser.add_argument('--prefix', default=None, type=str,
                        help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None,
                        type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DM", type=str, choices=[
                        'DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)


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


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' %
                     (mode, metric, step, metrics[metric]))


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, device, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(
        model, tp_answers, fn_answers, args, dataloader, query_name_dict, device)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(
            mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join(
                [mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join(
                [query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar(
            "_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics


def load_data(args, train_tasks, test_tasks, number_of_queries, number_of_entities):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    train_queries = pickle.load(
        open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(
        open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(
        open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(
        open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(
        open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in train_tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                                                                                    name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
        if name not in test_tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                                                                                    name, evaluate_union])]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]
        if number_of_queries >= 0:  # remove redundant train_queries
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                                                                                    name, evaluate_union])]
            remain_train_list = list(train_queries[query_structure])[
                :number_of_queries]
            del train_queries[query_structure]
            train_queries[query_structure].update(set(remain_train_list))
            remain_valid_list = list(valid_queries[query_structure])[
                :number_of_queries]
            del valid_queries[query_structure]
            valid_queries[query_structure].update(set(remain_valid_list))
            remain_test_list = list(test_queries[query_structure])[
                :number_of_queries]
            del test_queries[query_structure]
            test_queries[query_structure].update(set(remain_test_list))
        if number_of_entities >= 0:  # remove redundant entities
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([
                                                                                    name, evaluate_union])]
            if name == '1p':
                train_list = list(train_queries[query_structure])
                valid_list = list(valid_queries[query_structure])
                test_list = list(test_queries[query_structure])
                final_train_list = []
                final_valid_list = []
                final_test_list = []
                for i in range(len(train_list)):
                    if train_list[i][0] <= number_of_entities and max(list(train_answers[train_list[i]])) <= number_of_entities:
                        final_train_list.append(train_list[i])
                train_queries[query_structure] = train_queries[query_structure].intersection(
                    set(final_train_list))
                for i in range(len(valid_list)):
                    if valid_list[i][0] <= number_of_entities and max(list(valid_easy_answers[valid_list[i]].union(valid_hard_answers[valid_list[i]]))) <= number_of_entities:
                        final_valid_list.append(valid_list[i])
                valid_queries[query_structure] = valid_queries[query_structure].intersection(
                    set(final_valid_list))
                for i in range(len(test_list)):
                    if test_list[i][0] <= number_of_entities and max(list(test_easy_answers[test_list[i]].union(test_hard_answers[test_list[i]]))) <= number_of_entities:
                        final_test_list.append(test_list[i])
                train_queries[query_structure] = train_queries[query_structure].intersection(
                    set(final_test_list))
            if name == '2i':
                train_list = list(train_queries[query_structure])
                valid_list = list(valid_queries[query_structure])
                test_list = list(test_queries[query_structure])
                final_train_list = []
                final_valid_list = []
                final_test_list = []
                for i in range(len(train_list)):
                    if train_list[i][0][0] <= number_of_entities and train_list[i][1][0] <= number_of_entities and max(list(train_answers[train_list[i]])) <= number_of_entities:
                        final_train_list.append(train_list[i])
                train_queries[query_structure] = train_queries[query_structure].intersection(
                    set(final_train_list))
                for i in range(len(valid_list)):
                    if valid_list[i][0][0] <= number_of_entities and valid_list[i][1][0] <= number_of_entities and max(list(valid_easy_answers[valid_list[i]].union(valid_hard_answers[valid_list[i]]))) <= number_of_entities:
                        final_valid_list.append(valid_list[i])
                valid_queries[query_structure] = valid_queries[query_structure].intersection(
                    set(final_valid_list))
                for i in range(len(test_list)):
                    if test_list[i][0][0] <= number_of_entities and test_list[i][1][0] <= number_of_entities and max(list(test_easy_answers[test_list[i]].union(test_hard_answers[test_list[i]]))) <= number_of_entities:
                        final_test_list.append(test_list[i])
                train_queries[query_structure] = train_queries[query_structure].intersection(
                    set(final_test_list))

    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers


def main(args):
    set_global_seed(args.seed)
    train_tasks = args.train_tasks.split('.')
    test_tasks = args.test_tasks.split('.')
    for task in train_tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    for task in test_tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta' or args.geo == 'discrete', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split(
        '/')[-1], args.train_tasks, args.test_tasks, args.geo)
    if args.geo in ['box']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'discrete':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.hidden_dim)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        print(args.save_path)
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation
    if(args.number_of_entities >= 0 and args.number_of_entities < nentity):
        args.nentity = args.number_of_entities

    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries,\
        test_hard_answers, test_easy_answers = load_data(
        args, train_tasks, test_tasks, args.number_of_queries, args.number_of_entities)

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, args.nentity, nrelation,
                         args.negative_sample_size, train_answers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, args.nentity,
                             nrelation, args.negative_sample_size, train_answers),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None
    if args.do_test_in_train:
        train_easy_answers = defaultdict(set)
        train_queries = flatten_query(train_queries)
        train_dataloader = DataLoader(
            TestDataset(train_queries, args.nentity, args.nrelation, ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )
    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(
                query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    model = KGReasoning(
        nentity=args.nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        batch_size=args.batch_size,
        GMM_mode=eval_tuple(args.GMM_mode),
        use_cuda=args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode=eval_tuple(args.beta_mode),
        discrete_mode=eval_tuple(args.discrete_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.cuda))
        logging.info('Device use cuda: %s' % args.cuda)
    else:
        device = torch.device('cpu')
    if args.cuda:
        model = model.to(device)

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(
            args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    elif args.geo == 'discrete':
        logging.info('discrete mode = %s' % args.discrete_mode)
        logging.info('GMM mode = %s' % args.GMM_mode)
    logging.info('train_tasks = %s' % args.train_tasks)
    logging.info('test_tasks = %s' % args.test_tasks)
    logging.info('init_step = %d' % init_step)
    logging.info('lr = %d' % (args.learning_rate*10000))
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []
        # #Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = model.train_step(
                model, optimizer, train_path_iterator, args, step, device)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            if train_other_iterator is not None:
                log = model.train_step(
                    model, optimizer, train_other_iterator, args, step, device)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = model.train_step(
                    model, optimizer, train_path_iterator, args, step, device)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' %
                             (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_test_in_train:
                    logging.info('Evaluating on Train Dataset')
                    train_all_metrics = evaluate(model, train_easy_answers, train_answers,
                                                 args, train_dataloader, query_name_dict, device, 'Train', step, writer)
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers,
                                                 args, valid_dataloader, query_name_dict, device, 'Valid', step, writer)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers,
                                                args, test_dataloader, query_name_dict, device, 'Test', step, writer)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum(
                        [log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0
    if args.do_test_in_train:
        logging.info('Evaluating on Train Dataset...')
        train_all_metrics = evaluate(model, train_easy_answers, train_answers,
                                     args, train_dataloader, query_name_dict, device, 'Train', step, writer)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers,
                                    args, test_dataloader, query_name_dict, device, 'Test', step, writer)

    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())
