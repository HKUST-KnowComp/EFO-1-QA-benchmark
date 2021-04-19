import collections
import random
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from fol.foq import parse_foq_formula, gen_foq_meta_formula
from fol.sampler import load_data, read_indexing


def compare_depth_query(depth1, depth2, depth_dict,
                        projection_hard, reverse_hard,
                        start_point_num, query_num):
    start_point_set = random.sample(set(range(len(projection_hard))), start_point_num)
    print(start_point_set)
    stored_similarity = collections.defaultdict(dict)
    for ms1 in depth_dict[depth1][:query_num]:
        for ms2 in depth_dict[depth2][:query_num]:
            stored_similarity[ms1][ms2] = []
            sampler1 = parse_foq_formula(ms1)
            sampler2 = parse_foq_formula(ms2)
            for start_point in start_point_set:
                sampled_ans1 = sampler1.backward_sample(
                    projection_hard, reverse_hard, contain=True, keypoint=start_point)
                sampled_ans2 = sampler2.backward_sample(
                    projection_hard, reverse_hard, contain=True, keypoint=start_point)
                all_ans = sampled_ans1.union(sampled_ans2)
                shared_ans = sampled_ans1.intersection(sampled_ans2)
                similarity = len(shared_ans) / len(all_ans)
                stored_similarity[ms1][ms2].append(similarity)
    return stored_similarity


if __name__ == '__main__':
    stanford_data_path = '../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
    projection_none = {}
    reverse_proection_none = {}
    for i in all_entity_dict.values():
        projection_none[i] = collections.defaultdict(set)
        reverse_proection_none[i] = collections.defaultdict(set)
    projection_train, reverse_projection_train = load_data('../datasets_knowledge_embedding/FB15k-237/train.txt',
                                                           all_entity_dict, all_relation_dict, projection_none,
                                                           reverse_proection_none)
    depth_dict = collections.defaultdict(list)
    store_dict = collections.defaultdict(list)
    depth_low, depth_high = 3, 5
    start_point_num, query_num = 10, 100
    store_fold = '../data'
    for i in range(query_num):
        depth_low_q = gen_foq_meta_formula(max_depth=depth_low)
        depth_high_q = gen_foq_meta_formula(max_depth=depth_high)
        store_dict['query'].append(depth_low_q)
        store_dict['depth'].append(depth_low)
        store_dict['query'].append(depth_high_q)
        store_dict['depth'].append(depth_high)
        depth_dict[depth_low].append(depth_low_q)
        depth_dict[depth_high].append(depth_high_q)
    similarity = compare_depth_query(depth_low, depth_high, depth_dict,
                                     projection_train, reverse_projection_train,
                                     start_point_num=start_point_num, query_num=query_num)
    all_similarity = np.zeros(start_point_num*query_num**2)
    idx = 0
    query_similarity = np.zeros(query_num**2)
    for i in similarity:
        for j in similarity[i]:
            for k in similarity[i][j]:
                all_similarity[idx] = k
                query_similarity[int(idx/start_point_num)] += k/start_point_num
                idx += 1
    plt.hist(all_similarity, bins=20, density=False)
    plt.show()
    plt.hist(query_similarity, bins=20, density=True)
    plt.show()
    df = pd.DataFrame(data=store_dict)
    store_path = os.path.join(store_fold, f"meta_query_{depth_low}_{depth_high}.csv")
    df.to_csv(store_path, index=False)



