import collections
import os
import pandas as pd
import numpy as np
from fol import parse_formula, beta_query_v2
from typing import List


def count_ans(csv_name: List[str]):
    for name in csv_name:
        data = pd.read_csv(name)
        easy_ans_sets = data.easy_answer_set.map(
            lambda x: list(eval(x))).tolist()
        hard_ans_sets = data.hard_answer_set.map(
            lambda x: list(eval(x))).tolist()
        easy_ans_num = [len(easy_ans) for easy_ans in easy_ans_sets]
        hard_ans_num = [len(hard_ans) for hard_ans in hard_ans_sets]
        easy_ans_array, hard_ans_array = np.asarray(easy_ans_num), np.asarray(hard_ans_num)
        return easy_ans_array, hard_ans_array


if __name__ == "__main__":
    data_path = 'data/test_benchmark/FB15k-237-valid-foq'
    p_query = ['1p', '2p', '3p']
    csv_list = []
    for task in p_query:
        csv_list.append(os.path.join(data_path, f'test_{task}.csv'))
    easy_ans_array, hard_ans_array = count_ans(csv_list)
    print(np.mean(easy_ans_array), np.mean(hard_ans_array), np.bincount(hard_ans_array), max(hard_ans_array))





