from fol.sampler import *
from fol.foq_v2 import parse_formula
from fol.test_foq_v2 import beta_query_v2
import pandas as pd
from utils.util import load_rawdata_with_indexing


def sampling_stored(query_list, store_fold, num_queries, projs, rprojs, projs_hard, rprojs_hard, mode,
                    backward: bool = True):
    for query_name in query_list:
        query_structure = beta_query_v2[query_name]
        query_instance = parse_formula(query_structure)
        save_data = collections.defaultdict(list)
        i = int(0)
        while i < num_queries:
            if backward:
                full_ans = query_instance.backward_sample(projs_hard, rprojs_hard, cumulative=True)
            else:
                full_ans = query_instance.random_query(projs_hard, cumulative=True)
            if mode == 'test' or mode == 'valid':
                easy_ans = query_instance.deterministic_query(projs)
                hard_ans = full_ans - easy_ans
                if len(hard_ans) != 0:
                    save_data['query'].append(query_instance.dumps)
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
    data_path = 'data/FB15k-237-betae'
    rawdata_path = 'datasets_knowledge_embedding/FB15k-237'
    ent2id, rel2id, projection_train, reverse_projection_train, projection_valid, reverse_projection_valid, \
    projection_test, reverse_projection_test = load_rawdata_with_indexing(data_path, rawdata_path)
    NewLook_Query = ['2D', '3D', 'Dp']
    sampling_stored(NewLook_Query, data_path, 149689, None, None,
                    projection_train, reverse_projection_train, 'train', True)
    sampling_stored(NewLook_Query, data_path, 5000, projection_train, reverse_projection_train, projection_valid,
                    reverse_projection_valid, 'valid', True)
    sampling_stored(NewLook_Query, data_path, 5000, projection_valid, reverse_projection_valid, projection_test,
                    reverse_projection_test, 'test', True)


