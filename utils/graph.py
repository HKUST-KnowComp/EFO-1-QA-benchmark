import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
from fol import beta_query_v2, parse_formula, beta_query, parse_foq_formula

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


def compare_loss(path, path2, choose_len=None):
    data_file = os.path.join(path, 'train.csv')
    data_2 = os.path.join(path2, 'beta_train.csv')
    df, df2 = pd.read_csv(data_file), pd.read_csv(data_2)
    loss, loss2 = np.asarray(df['loss']), np.asarray(df2['loss'])
    step= np.asarray(df['step'])
    minlen = min(len(loss), len(loss2))
    if choose_len:
        loss = loss[:choose_len]
        loss2 = loss2[:choose_len]
        step = step[:choose_len]
    if len(loss) > minlen:
        loss = loss[:minlen]
    else:
        loss2 = loss2[:minlen]

    compare = np.log(loss) - np.log(loss2)
    plt.plot(step, compare)
    plt.plot(step, np.zeros_like(compare), color='r')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()


def log_all_metrics(path, step, mode, log_meta_formula=beta_query_v2.values()):
    log = collections.defaultdict(lambda: collections.defaultdict(float))

    for meta_formula in log_meta_formula:
        # if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_formula(meta_formula)
        foq_formula = foq_instance.formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in log_meta_formula]
    for metric in log:
        averaged_metric[metric] = \
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula])/len(averaged_my_formula)
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)
    print(averaged_metric)


def log_old_metrics(path, step, mode, log_meta_formula=beta_query.values()):
    log = collections.defaultdict(lambda: collections.defaultdict(float))

    for meta_formula in log_meta_formula:
        # if meta_formula != 'p(e)|p(e)' and meta_formula != 'p(p(e)|p(e))':
        foq_instance = parse_foq_formula(meta_formula)
        foq_formula = foq_instance.meta_formula
        data_file = os.path.join(path, f'eval_{mode}_{foq_formula}.csv')
        df = pd.read_csv(data_file)
        step_range = np.asarray(df['step'])
        step_index = np.where(step_range == step)[0]
        for metric in df.columns:
            if metric != 'step':
                log[metric][foq_formula] = df[metric][step_index].values[0]
    averaged_metric = {}
    averaged_my_formula = [parse_foq_formula(formula).meta_formula for formula in log_meta_formula]
    for metric in log:
        averaged_metric[metric] = \
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula])/len(averaged_my_formula)
    all_data = pd.DataFrame.from_dict(log)
    all_data.to_csv(os.path.join(path, f'eval_{mode}_{step}_average.csv'))
    print(all_data)
    print(averaged_metric)


def read_beta_log(path, mode='test', chosen_step=None, averaged_meta_formula=beta_query_v2.values()):
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
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    valid_log[step][metric][foq_formula] = score
                    beta_valid[foq_formula][metric].append(score)
            elif line[29:34] == 'Test ' and line[34:38] != 'info':
                info = line[34:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    test_log[step][metric][foq_formula] = score
                    beta_test[foq_formula][metric].append(score)
    train_data = pd.DataFrame.from_dict(train_log)
    train_data.to_csv(os.path.join(path, 'beta_train.csv'))
    # print(pd.DataFrame.from_dict(valid_log[chosen_step]))
    for step in eval(f'{mode}_log'):
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'beta_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'beta_test_{step}.csv'))
    if chosen_step is not None:
        print(pd.DataFrame.from_dict(test_log[chosen_step]))
    else:
        print(test_data)
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in averaged_meta_formula]
    for metric in test_log[15000]:
        if chosen_step is not None:
            averaged_metric[metric] = sum([test_log[chosen_step][metric][foq_formula]
                                           for foq_formula in averaged_my_formula]) / len(averaged_meta_formula)
    print(averaged_metric)
    return train_log, beta_valid, beta_test


def read_logic_log(path, mode='test', chosen_step=None, averaged_meta_formula=beta_query_v2.values()):
    train_log = collections.defaultdict(lambda: collections.defaultdict(float))
    valid_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    test_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    logic_valid = collections.defaultdict(lambda: collections.defaultdict(list))
    logic_test = collections.defaultdict(lambda: collections.defaultdict(list))
    beta_log_path = os.path.join(path, 'train.log')
    with open(beta_log_path, 'r') as f:
        for line in f.readlines():
            if line[29:50] == 'Training average loss':
                info = line[58:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:61] == 'Training average positive_sample':
                info = line[75:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:61] == 'Training average negative_sample':
                info = line[75:]
                step,  score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['negative_loss'][step] = score
            elif line[29:35] == 'Valid ':
                info = line[35:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    valid_log[step][metric][foq_formula] = score
                    logic_valid[foq_formula][metric].append(score)
            elif line[29:34] == 'Test ' and line[34:38] != 'info':
                info = line[34:].split(' ')
                beta_name, metric, step, score = info[0], info[1], eval(info[4][:-1]), eval(info[5])
                if beta_name in beta_query_v2:
                    foq_instance = parse_formula(beta_query_v2[beta_name])
                    foq_formula = foq_instance.formula
                    test_log[step][metric][foq_formula] = score
                    logic_test[foq_formula][metric].append(score)
    train_data = pd.DataFrame.from_dict(train_log)
    train_data.to_csv(os.path.join(path, 'beta_train.csv'))
    # print(pd.DataFrame.from_dict(valid_log[chosen_step]))
    for step in eval(f'{mode}_log'):
        valid_data = pd.DataFrame.from_dict(valid_log[step])
        valid_data.to_csv(os.path.join(path, f'logic_valid_{step}.csv'))
        test_data = pd.DataFrame.from_dict(test_log[step])
        test_data.to_csv(os.path.join(path, f'logic_test_{step}.csv'))
    if chosen_step is not None:
        print(pd.DataFrame.from_dict(test_log[chosen_step]))
    else:
        print(test_data)
    averaged_metric = {}
    averaged_my_formula = [parse_formula(formula).formula for formula in averaged_meta_formula]
    for metric in test_log[15000]:
        if chosen_step is not None:
            averaged_metric[metric] = sum([test_log[chosen_step][metric][foq_formula]
                                           for foq_formula in averaged_my_formula]) / len(averaged_meta_formula)
    print(averaged_metric)
    return train_log, logic_valid, logic_test


def plot_comparison(beta_log, my_log, all_formula):
    # metric in ['MRR', 'HITS1', 'HITS3', 'HITS10']:
    for metric in ['MRR']:
        for meta_formula in all_formula:
            foq_instance = parse_formula(beta_query_v2[meta_formula])
            foq_formula = foq_instance.formula
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
            foq_instance = parse_formula(beta_query_v2[meta_formula])
            foq_formula = foq_instance.formula
            df = pd.read_csv(os.path.join(path, f'eval_{mode}_{foq_formula}.csv'))
            for metric in df.columns:
                if metric != 'step' and metric != 'num_queries':
                    for i in range(len(df[metric])):
                        eval(f'my_{mode}')[foq_formula][metric].append(df[metric][i])
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '3p'], mode)
        # plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['2i', '3i'], mode)
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['1p', '2p', '2i'])
        plot_comparison(eval(f'beta_{mode}'), eval(f'my_{mode}'), ['3p', '3i'])


def log_benchmark(folder_path, id_filename, id_list):
    all_formula = pd.read_csv(id_filename)
    all_log = collections.defaultdict(lambda: collections.defaultdict(float))
    for task_id in id_list:
        id_str = str(task_id)
        id_str = '0' * (4 - len(id_str)) + id_str
        single_log = pd.read_csv(os.path.join(folder_path, f'eval_type{id_str}.csv'))
        real_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
        for metric in single_log.columns:
            all_log[task_id][metric] = single_log[metric]
    data_all_log = pd.DataFrame.from_dict(all_log)
    data_all_log.to_csv(os.path.join(folder_path, 'all_formula_log.csv'))
    return all_log










box_query_v2 = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
}

check_query = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2in': '(i,(p,(e)),(n,(p,(e))))',
    '3in': '(i,(p,(e)),(p,(e)),(n,(p,(e))))',
    'inp': '(p,(i,(p,(e)),(n,(p,(e)))))',
    'pin': '(i,(p,(p,(e))),(n,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
    '2u-DM': '(n,(i,(n,(p,(e))),(n,(p,(e)))))',
    'up-DM': '(p,(n,(i,(n,(p,(e))),(n,(p,(e))))))',
}
DNF_query = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2in': '(i,(p,(e)),(n,(p,(e))))',
    '3in': '(i,(p,(e)),(p,(e)),(n,(p,(e))))',
    'inp': '(p,(i,(p,(e)),(n,(p,(e)))))',
    'pin': '(i,(p,(p,(e))),(n,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
}
# print_loss(graph_path)
'''
test_step = 450000
test_path = "/home/hyin/DiscreteMeasureReasoning/log/newdev/Logic-unbounded210813.22:26:062c614d51/"
old_path = "/home/hyin/DiscreteMeasureReasoning/log/newdev/Logic-unbounded210813.21:19:26aaf6eebf/"
# test_path = "/home/hyin/DiscreteMeasureReasoning/log/dev/default210705.14:43:26fba267b0/"
logic_path = "/data/zwanggc/Logic-unbounded210813.22:24:17607989e2/"
#compare_loss(test_path, test_path, choose_len=3000)
log_all_metrics(test_path, test_step, 'test', log_meta_formula=check_query.values())
log_all_metrics(old_path, test_step, 'test', log_meta_formula=check_query.values())
'''

benchmark_path = ''
log_benchmark(benchmark_path,'data/generated_formula_anchor_node=3.csv', list(range(0, 464)))
#log_old_metrics(old_path, test_step, 'test')
# train_all, valid_all, test_all = read_beta_log('../download_log/full/')
# train_part, valid_part, test_part = read_logic_log(logic_path, 'test', test_step, averaged_meta_formula=DNF_query.values())


#comparison('../download_log/1p.2p.2i/', ['1p', '2p', '2i'])

