import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
from fol import beta_query_v2, parse_formula, beta_query, parse_foq_formula
from data_helper import all_normal_form

beta_step = [15000 * i for i in range(1, 21)] + [360000, 420000, 450000]
beta_valid_step = [15000 * i for i in range(1, 21)] + [360000, 420000]

step_dict = {i: beta_step[i] for i in range(len(beta_step))}
inverse_step_dict = {beta_step[i]: i for i in range(len(beta_step))}

all_metrics = ['MRR', 'HITS1', 'HITS3', 'HITS10', 'retrieval_accuracy']
model_supportform_dict = {
    'Beta': ['DeMorgan', 'DeMorgan+MultiI', 'DNF+MultiIU'],
    'Logic': ['DeMorgan', 'DeMorgan+MultiI', 'DNF+MultiIU'],
    'NewLook': ['DNF+MultiIUD']
}
model_compareform_dict = {
    'Beta': ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'DNF+MultiIU'],
    'Logic': ['original', 'DeMorgan', 'DeMorgan+MultiI', 'DNF', 'DNF+MultiIU'],
    'NewLook': ['original', 'DNF', 'diff', 'DNF+diff', 'DNF+MultiIU', 'DNF+MultiIUD']
}

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
    step = np.asarray(df['step'])
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
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula]) / len(averaged_my_formula)
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
            sum([log[metric][foq_formula] for foq_formula in averaged_my_formula]) / len(averaged_my_formula)
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
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:54] == 'Training average positive':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:54] == 'Training average negative':
                info = line[75:]
                step, score = info.split(':')
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
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['loss'][step] = score
            elif line[29:61] == 'Training average positive_sample':
                info = line[75:]
                step, score = info.split(':')
                step, score = eval(step), eval(score)
                train_log['positive_loss'][step] = score
            elif line[29:61] == 'Training average negative_sample':
                info = line[75:]
                step, score = info.split(':')
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


def log_benchmark(folder_path, id_list, percentage=False):
    all_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for task_id in id_list:
        id_str = str(task_id)
        id_str = '0' * (4 - len(id_str)) + id_str
        # real_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
        if os.path.exists(os.path.join(folder_path, f'eval_type{id_str}.csv')):
            single_log = pd.read_csv(os.path.join(folder_path, f'eval_type{id_str}.csv'))
            index2metrics = single_log['Unnamed: 0']
            for normal_form in single_log.columns:
                if normal_form != 'Unnamed: 0':
                    for index in range(len(single_log[normal_form])):
                        if percentage and index2metrics[index] != 'num_queries':
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index] * 100
                        else:
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index]
    for metric in all_log:
        data_metric = pd.DataFrame.from_dict(all_log[metric])
        data_metric.to_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
    return all_log


def normal_form_comparison(folder_path, form1, form2, metrics, save_csv=False, percentage=False):
    all_formula = pd.read_csv('data/generated_formula_anchor_node=3.csv')
    unequal_task = set()
    form1_log, form2_log = collections.defaultdict(lambda: collections.defaultdict(float)), \
        collections.defaultdict(lambda: collections.defaultdict(float))
    comparison_log = collections.defaultdict(list)
    for metric in metrics:
        metric_logging = pd.read_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
        index2taskid = metric_logging['Unnamed: 0']
        for index in range(len(index2taskid)):
            taskid = index2taskid[index]
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
            formula1, formula2 = all_formula[form1][formula_index], all_formula[form2][formula_index]
            score1, score2 = metric_logging[form1][index], metric_logging[form2][index]
            if formula1 != formula2 and str(score1) != 'nan' and str(score2) != 'nan':
                # what if two scores are same
                if taskid not in unequal_task:
                    assert metric == metrics[0]
                    unequal_task.add(taskid)
                form1_log[metric][taskid], form2_log[metric][taskid] = score1, score2
    if len(unequal_task) > 0:
        for metric in metrics:
            averaged1, averaged2 = sum(form1_log[metric][taskid] for taskid in form1_log[metric]) / \
                                   len(form1_log[metric]), \
                                   sum(form2_log[metric][taskid] for taskid in form2_log[metric]) / \
                                   len(form2_log[metric])
            comparison_log[metric] = [averaged1, averaged2]
    else:
        for metric in metrics:
            comparison_log[metric] = [0, 0]
    comparison_log['different_queries'] = [len(unequal_task), len(unequal_task)]
    if save_csv:
        compare_taskid = {}
        for metric in metrics:
            compare_taskid[f'{form1}_{metric}'] = form1_log[metric]
        compare_taskid[f'{form1}_formula'] = {}
        compare_taskid[f'{form2}_formula'] = {}
        for taskid in unequal_task:
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
            formula1, formula2 = all_formula[form1][formula_index], all_formula[form2][formula_index]
            compare_taskid[f'{form1}_formula'][taskid] = formula1
            compare_taskid[f'{form2}_formula'][taskid] = formula2
        data = pd.DataFrame.from_dict(compare_taskid)
        data.to_csv(os.path.join(folder_path, f'compare_detail_{form1}_{form2}.csv'))

    '''
    df = pd.DataFrame.from_dict(comparison_log)
    df.to_csv(os.path.join(folder_path, f'compare_{form1}_{form2}.csv'))
    '''
    return comparison_log


def compare_all_form(folder_path, form_list, metrics, save_csv=False):
    difference_mrr = collections.defaultdict(lambda: collections.defaultdict(list))
    difference_number = collections.defaultdict(lambda: collections.defaultdict(int))
    n = len(form_list)
    for i in range(n):
        for j in range(i + 1, n):
            comparison_log = normal_form_comparison(folder_path, form_list[i], form_list[j], metrics, save_csv)
            difference_mrr[form_list[j]][form_list[i]] = comparison_log['MRR']
            difference_number[form_list[j]][form_list[i]] = comparison_log['different_queries'][0]
    print(difference_number)
    print(difference_mrr)
    dm, dn = pd.DataFrame.from_dict(difference_mrr), pd.DataFrame.from_dict(difference_number)
    dm.to_csv(os.path.join(folder_path, f'allmrr_compare.csv'))
    dn.to_csv(os.path.join(folder_path, f'alllength_compare.csv'))


def log_benchmark_depth_anchornode(folder_path, support_normal_forms, metrics):
    all_formula = pd.read_csv('data/generated_formula_anchor_node=3.csv')
    all_logging = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    averaged_split = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    averaged_all = collections.defaultdict(lambda: collections.defaultdict(float))
    for normal_form in support_normal_forms:
        for i in range(1, 4):
            for j in range(1, 4):
                for metric in metrics:
                    averaged_split[normal_form][(i, j)][metric] = 0
    for metric in metrics:
        metric_logging = pd.read_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
        index2taskid = metric_logging['Unnamed: 0']
        for index in range(len(index2taskid)):
            taskid = index2taskid[index]
            id_str = '0' * (4 - len(str(taskid))) + str(taskid)
            formula_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
            depth = all_formula['original_depth'][formula_index]
            anchornode_num = all_formula['num_anchor_nodes'][formula_index]
            for normal_form in support_normal_forms:
                query_scores = metric_logging.loc[index][normal_form]
                all_logging[normal_form][(anchornode_num, depth)][metric].append(query_scores)
    all_number = sum(len(all_logging[support_normal_forms[0]][key][metrics[0]])
                     for key in all_logging[support_normal_forms[0]])
    assert all_number == 251  # all query type are included
    for normal_form in support_normal_forms:
        for key in all_logging[normal_form]:
            for metric in metrics:
                averaged_split[normal_form][key][metric] = sum(all_logging[normal_form][key][metric])\
                                                           / len(all_logging[normal_form][key][metric])
    for normal_form in support_normal_forms:
        for metric in metrics:
            averaged_all[normal_form][metric] = sum(sum(all_logging[normal_form][key][metric])
                                                    for key in all_logging[normal_form])
            averaged_all[normal_form][metric] /= 251
            averaged_split[normal_form]['average'][metric] = averaged_all[normal_form][metric]
        df = pd.DataFrame.from_dict(averaged_split[normal_form])
        df.to_csv(os.path.join(folder_path, f'anchornode_depth_of_{normal_form}.csv'))
    print(averaged_all)
    return averaged_split


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

beta_path = '/home/zwanggc/DiscreteMeasureReasoning/benchmark_log/benchmark_FB15k-237/Beta_full210822.00:58:08058bb673'
NLK_path = "/home/zwanggc/DiscreteMeasureReasoning/benchmark_log/benchmark_FB15k-237/NLK_full210822.11:13:14cdab5a51"
Logic_path = "/home/hyin/DiscreteMeasureReasoning/benchmark_log/benchmark_FB15k-237/Logic_full210824.00:02:41898e39b7"
Box_path = "/home/zwanggc/DiscreteMeasureReasoning/benchmark_log/benchmark_FB15k-237/Box_full210822.00:56:4448dc3a71"
id_file = 'data/generated_formula_anchor_node=3.csv'
log_benchmark(Logic_path, list(range(0, 464)), percentage=True)
# compare_all_form(Box_path, all_normal_form, all_metrics)
compare_all_form(Logic_path, model_compareform_dict['Logic'], metrics=all_metrics, save_csv=True)
log_benchmark_depth_anchornode(Logic_path, model_supportform_dict['Logic'], all_metrics)

# log_old_metrics(old_path, test_step, 'test')
# train_all, valid_all, test_all = read_beta_log('../download_log/full/')
# train_part, valid_part, test_part = read_logic_log(logic_path, 'test', test_step, averaged_meta_formula=DNF_query.values())


# comparison('../download_log/1p.2p.2i/', ['1p', '2p', '2i'])
