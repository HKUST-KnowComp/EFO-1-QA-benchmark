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


def log_all_metrics(path, step, mode, averaged_meta_formula=beta_query.values()):
    log = collections.defaultdict(lambda: collections.defaultdict(float))

    for meta_formula in beta_query.values():
        # if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_foq_formula(meta_formula)
        foq_formula = foq_instance.meta_formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step' and metric != 'num_queries':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    averaged_metric = {}
    averaged_my_formula = [parse_foq_formula(formula).meta_formula for formula in averaged_meta_formula]
    for metric in log:
        averaged_metric[metric] = \
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula])/len(averaged_my_formula)
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)
    print(averaged_metric)


def read_beta_log(path, mode='test', chosen_step=None, averaged_meta_formula=beta_query.values()):
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
    # print(pd.DataFrame.from_dict(valid_log[chosen_step]))
    for step in f'{mode}_log':
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'beta_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'beta_test_{step}.csv'))
    if chosen_step is not None:
        print(pd.DataFrame.from_dict(test_log[chosen_step]))
    else:
        print(test_data)
    averaged_metric = {}
    averaged_my_formula = [parse_foq_formula(formula).meta_formula for formula in averaged_meta_formula]
    for metric in test_log[15000]:
        if chosen_step is not None:
            averaged_metric[metric] = sum([test_log[chosen_step][metric][foq_formula]
                                           for foq_formula in averaged_my_formula]) / len(averaged_meta_formula)
    print(averaged_metric)
    return train_log, beta_valid, beta_test


def plot_comparison(beta_log, my_log, all_formula):
    # metric in ['MRR', 'HITS1', 'HITS3', 'HITS10']:
    for metric in ['MRR']:
        for meta_formula in all_formula:
            foq_instance = parse_foq_formula(beta_query[meta_formula])
            foq_formula = foq_instance.meta_formula
            beta_score = np.asarray(beta_log[foq_formula][metric])
            my_score = np.asarray(my_log[foq_formula][metric])
            n = len(my_score)
            beta_plot_step = np.asarray(beta_step)[:n]
            plt.plot(beta_plot_step, beta_score[:n], color='red', label=f'{meta_formula}_beta')
            plt.plot(beta_plot_step, my_score, linestyle=':', color='blue', label=f'{meta_formula}_ours')
        plt.title(all_formula)
        plt.legend()
        plt.show()


def comparison(path, all_meta_formula):
    our_train = pd.read_csv(os.path.join(path, 'train.csv'))
    my_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    my_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_train, beta_valid, beta_test = read_beta_log(path)
    for mode in ['valid', 'test']:
        for meta_formula in all_meta_formula:
            foq_instance = parse_foq_formula(beta_query[meta_formula])
            foq_formula = foq_instance.meta_formula
            df = pd.read_csv(os.path.join(path, f'eval_{mode}_{foq_formula}.csv'))
            for metric in df.columns:
                if metric != 'step' and metric != 'num_queries':
                    for i in range(len(df[metric])):
                        eval(f'my_{mode}')[foq_formula][metric].append(df[metric][i])
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '3p'], mode)
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['2i', '3i'], mode)
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '2i'])
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['3p', '3i'])

















# print_loss(graph_path)
beta_except_3i = [beta_query[formula] for formula in beta_query.keys() if formula != '3i' and formula != '3in']
test_step = 45000
test_path = "../log/dev/default_except3i210706.11:07:0086b2bdb6/"
# test_path = "/home/hyin/DiscreteMeasureReasoning/log/dev/default_1p2i210705.14:44:149b72aa8a/"
log_all_metrics(test_path, test_step, 'valid', beta_except_3i)
# train_all, valid_all, test_all = read_beta_log('../download_log/full/')
train_part, valid_part, test_part = read_beta_log(test_path, 'valid', test_step, beta_except_3i)


#comparison('../download_log/1p.2p.2i/', ['1p', '2p', '2i'])

