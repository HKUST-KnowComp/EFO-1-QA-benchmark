import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
graph_path = '../download_log/'


def print_loss():
    data_file = os.path.join(graph_path, 'train.csv')
    df = pd.read_csv(data_file)
    loss = np.asarray(df['loss'])
    step = np.asarray(df['step'])
    loss = np.log(loss)
    plt.plot(step, loss)
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()


print_loss()


