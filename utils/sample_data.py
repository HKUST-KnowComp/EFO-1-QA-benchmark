import collections
import os
import pandas as pd
from fol import parse_formula, beta_query_v2
from utils.util import load_data_with_indexing


def sampling_stored(query_list, store_fold, num_queries, projs, projs_hard, rprojs_hard, mode,
                    backward: bool = True):
    for query_name in query_list:
        query_structure = beta_query_v2[query_name]
        query_instance = parse_formula(query_structure)
        save_data = collections.defaultdict(list)
        all_query = set()
        i = int(0)
        while i < num_queries:
            if backward:
                full_ans = query_instance.backward_sample(projs_hard, rprojs_hard)
            else:
                full_ans = query_instance.random_query(projs_hard)
            assert query_instance.deterministic_query(projs_hard) == full_ans
            easy_ans = query_instance.deterministic_query(projs)
            hard_ans = full_ans - easy_ans
            if len(hard_ans) > 100 or len(hard_ans) == 0 or query_instance.dumps in all_query:
                query_instance.lift()
                continue
            if mode == 'test' or mode == 'valid':
                save_data['query'].append(query_instance.dumps)
                all_query.add(query_instance.dumps)
                save_data['id'].append(i)
                i += 1
                save_data['easy_answer_set'].append(easy_ans)
                save_data['hard_answer_set'].append(hard_ans)
            elif mode == 'train':
                if len(full_ans) > 0:
                    save_data['query'].append(query_instance.dumps)
                    save_data['id'].append(i)
                    i += 1
                    save_data['answer_set'].append(full_ans)
            query_instance.lift()
            if i % 1000 == 0:
                print(f'{mode} split of {query_name} have sampled {i} queries')
        df = pd.DataFrame(data=save_data)
        storation_path = os.path.join(store_fold, f"{mode}_{query_name}.csv")
        df.to_csv(storation_path, index=False)


if __name__ == "__main__":
    data_path = 'data/test_benchmark/FB15k-237-valid-foq-22803'
    read_data_path = 'data/FB15k-237-betae'
    ent2id, rel2id, projection_train, reverse_projection_train, projection_valid, reverse_projection_valid, \
        projection_test, reverse_projection_test = load_data_with_indexing(read_data_path)
    NewLook_Query = ['2D', '3D', 'Dp']
    p_Query = ['1p', '2p', '3p']
    generalize_query = ['4p', '4i']
    '''
    sampling_stored(p_Query, data_path, 10798, None,
                    projection_train, reverse_projection_train, 'train', True)
    sampling_stored(p_Query, data_path, 4000, projection_train, projection_valid,
                    reverse_projection_valid, 'valid', True)
                   
    sampling_stored(['1p'], data_path, 22804, projection_valid, projection_test,
                    reverse_projection_test, 'test', True)
    '''
    sampling_stored(generalize_query, data_path, 5000, projection_valid, projection_test,
                    reverse_projection_test, 'test', True)



