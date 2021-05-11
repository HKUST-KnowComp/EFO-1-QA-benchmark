import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
from fol.base import beta_query
from fol.foq import parse_foq_formula
graph_path = '../download_log/'


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


#print_loss(graph_path)
test_path = 'log/dev/copy'
log_all_metrics(test_path, 360000, 'test')


