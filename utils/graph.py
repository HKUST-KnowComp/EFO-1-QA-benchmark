import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
from fol.base import beta_query
from fol.foq import parse_foq_formula
graph_path = '../download_log/'

beta_step = [15000*i for i in range(1, 21)] + [360000, 420000, 450000]
beta_valid_step = [15000*i for i in range(1, 21)] + [360000, 420000]

step_dict = {i: beta_step[i] for i in range(len(beta_step))}
inverse_step_dict = {beta_step[i]: i for i in range(len(beta_step))}

def print_loss(path):
    data_file = os.path.join(path, 'train.csv')
    df = pd.read_csv(data_file)
    loss = np.asarray(df['loss'])
    step = np.asarray(df['step'])
    loss = np.log(loss)
    plt.plot(step, loss)
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()


def log_all_metrics(path, step, mode):
    log = collections.defaultdict(lambda: collections.defaultdict(float))
    for meta_formula in beta_query.values():
        #if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_foq_formula(meta_formula)
        foq_formula = foq_instance.meta_formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step' and metric != 'num_queries':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)


def read_beta_log(path):
    train_log = collections.defaultdict(lambda: collections.defaultdict(float))
    valid_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    test_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    beta_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_log_path = os.path.join(path, 'train.log')
    with open(beta_log_path, 'r') as f:
        for line in f.readlines():
            if line[29:50] == 'Training average loss':
                info = line[58:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:54] == 'Training average positive':
                info = line[75:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:54] == 'Training average negative':
                info = line[75:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['negative_loss'][step] = score
            elif line[29:35] == 'Valid ':
                info = line[35:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query:
                    foq_instance = parse_foq_formula(beta_query[beta_name])
                    foq_formula = foq_instance.meta_formula
                    valid_log[step][metric][foq_formula] = score
                    beta_valid[foq_formula][metric].append(score)
            elif line[29:34] == 'Test ' and line[34:38] != 'info':
                info = line[34:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query:
                    foq_instance = parse_foq_formula(beta_query[beta_name])
                    foq_formula = foq_instance.meta_formula
                    test_log[step][metric][foq_formula] = score
                    beta_test[foq_formula][metric].append(score)
    train_data = pd.DataFrame.from_dict(train_log)
    train_data.to_csv(os.path.join(path, 'beta_train.csv'))
    for step in test_log:
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'beta_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'beta_test_{step}.csv'))
    return train_log, beta_valid, beta_test


def plot_comparison(beta_log, my_log, all_formula, mode):
    # metric in ['MRR', 'HITS1', 'HITS3', 'HITS10']:
    for metric in ['MRR']:
        for meta_formula in all_formula:
            foq_instance = parse_foq_formula(beta_query[meta_formula])
            foq_formula = foq_instance.meta_formula
            beta_score = np.asarray(beta_log[foq_formula][metric])
            my_score = np.asarray(my_log[foq_formula][metric])
            if mode == 'valid':
                plt.plot(np.asarray(beta_valid_step), beta_score, color='red', label=f'{meta_formula}_beta')
                plt.plot(np.asarray(beta_valid_step), my_score, linestyle=':', color='blue',
                         label=f'{meta_formula}_ours')
            else:
                plt.plot(np.asarray(beta_step), beta_score, color='red', label=f'{meta_formula}_beta')
                plt.plot(np.asarray(beta_step), my_score, linestyle=':', color='blue', label=f'{meta_formula}_ours')
        plt.title(all_formula)
        plt.legend()
        plt.show()


def comparison(path):
    our_train = pd.read_csv(os.path.join(path, 'train.csv'))
    my_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    my_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_train, beta_valid, beta_test = read_beta_log(path)
    for mode in ['valid', 'test']:
        for meta_formula in beta_query.values():
            foq_instance = parse_foq_formula(meta_formula)
            foq_formula = foq_instance.meta_formula
            df = pd.read_csv(os.path.join(path, f'eval_{mode}_{foq_formula}.csv'))
            for metric in df.columns:
                if metric != 'step' and metric != 'num_queries':
                    for i in range(len(beta_step)):
                        if mode == 'test' or i != 22:
                            eval(f'my_{mode}')[foq_formula][metric].append(df[metric][i])
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '3p'], mode)
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['2i', '3i'], mode)


















# print_loss(graph_path)
test_path = 'log/dev/copy'
# log_all_metrics(test_path, 120000, 'test')
read_beta_log('../download_log/full/')
comparison('../download_log/full/')

