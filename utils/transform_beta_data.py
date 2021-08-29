import collections
import os
import pickle
from fol.sampler import *
from fol.foq import parse_foq_formula
from fol.foq_v2 import parse_formula
from fol import beta_query_v2
from utils.util import load_rawdata_with_indexing

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }


def transform_query(query, meta_formula):
    if meta_formula == '1p':
        e, r = query[0], query[1][0]
        new_query = f"[{r}]({{{e}}})"
    elif meta_formula == '2p':
        e1, r1, r2 = query[0], query[1][0], query[1][1]
        new_query = f"[{r2}]([{r1}]({{{e1}}}))"
    elif meta_formula == '3p':
        e1, r1, r2, r3 = query[0], query[1][0], query[1][1], query[1][2]
        new_query = f"[{r3}]([{r2}]([{r1}]({{{e1}}})))"
    elif meta_formula == '2i':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f"[{r1}]({{{e1}}})&[{r2}]({{{e2}}})"
    elif meta_formula == '3i':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        new_query = f"[{r1}]({{{e1}}})&[{r2}]({{{e2}}})&[{r3}]({{{e3}}})"
    elif meta_formula == 'ip':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f"[{r3}]([{r1}]({{{e1}}})&[{r2}]({{{e2}}}))"
    elif meta_formula == 'pi':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f"[{r2}]([{r1}]({{{e1}}}))&[{r3}]({{{e2}}})"
    elif meta_formula == '2in':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f"[{r1}]({{{e1}}})-[{r2}]({{{e2}}})"
    elif meta_formula == '3in':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        new_query = f"[{r1}]({{{e1}}})&[{r2}]({{{e2}}})-[{r3}]({{{e3}}})"
    elif meta_formula == 'inp':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f"[{r3}]([{r1}]({{{e1}}})-[{r2}]({{{e2}}}))"
    elif meta_formula == 'pin':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f"[{r2}]([{r1}]({{{e1}}}))-[{r3}]({{{e2}}})"
    elif meta_formula == 'pni':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f"[{r3}]({{{e2}}})-[{r2}]([{r1}]({{{e1}}}))"
    elif meta_formula == '2u-DNF':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f"[{r1}]({{{e1}}})|[{r2}]({{{e2}}})"
    elif meta_formula == 'up-DNF':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f"[{r3}]([{r1}]({{{e1}}})|[{r2}]({{{e2}}}))"
    elif meta_formula == '2u-DM':
        e1, e2, r1, r2 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0]
        new_query = f"[{r1}]({{{e1}}})|[{r2}]({{{e2}}})"
    elif meta_formula == 'up-DM':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][1]
        new_query = f"[{r3}]([{r1}]({{{e1}}})|[{r2}]({{{e2}}}))"
    else:
        new_query = None
        print('not valid name!')
    return new_query


def transform_json_query(query, meta_formula):
    if meta_formula == '1p':
        e, r = query[0], query[1][0]
        new_query = f'{{"o": "p", "a": [[{r}], {{"o": "e", "a": [{e}]}}]}}'
    elif meta_formula == '2p':
        e1, r1, r2 = query[0], query[1][0], query[1][1]
        new_query = f'{{"o": "p", "a": [[{r2}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}'
    elif meta_formula == '3p':
        e1, r1, r2, r3 = query[0], query[1][0], query[1][1], query[1][2]
        new_query = f'{{"o": "p", "a": [[{r3}], {{"o": "p", "a": [[{r2}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}]}}'
    elif meta_formula == '2i':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}]}}'
    elif meta_formula == '3i':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}, {{"o": "p", "a": [[{r3}], {{"o": "e", "a": [{e3}]}}]}}]}}'
    elif meta_formula == 'ip':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f'{{"o": "p", "a": [[{r3}], {{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}]}}]}}'
    elif meta_formula == 'pi':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r2}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}, {{"o": "p", "a": [[{r3}], {{"o": "e", "a": [{e2}]}}]}}]}}'
    elif meta_formula == '2in':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "n", "a": {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}}}]}}'
    elif meta_formula == '3in':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}, {{"o": "n", "a": {{"o": "p", "a": [[{r3}], {{"o": "e", "a": [{e3}]}}]}}}}]}}'
    elif meta_formula == 'inp':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f'{{"o": "p", "a": [[{r3}], {{"o": "i", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "n", "a": {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}}}]}}]}}'
    elif meta_formula == 'pin':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "p", "a": [[{r2}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}, {{"o": "n", "a": {{"o": "p", "a": [[{r3}], {{"o": "e", "a": [{e2}]}}]}}}}]}}'
    elif meta_formula == 'pni':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        new_query = f'{{"o": "i", "a": [{{"o": "n", "a": {{"o": "p", "a": [[{r2}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}}}, {{"o": "p", "a": [[{r3}], {{"o": "e", "a": [{e2}]}}]}}]}}'
    elif meta_formula == '2u-DNF':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        new_query = f'{{"o": "u", "a": [{{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}, {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}]}}'
    elif meta_formula == 'up-DNF':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        new_query = f'{{"o": "u", "a": [{{"o": "p", "a": [[{r3}], {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}]}}, {{"o": "p", "a": [[{r3}], {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}]}}]}}'
    elif meta_formula == '2u-DM':
        e1, e2, r1, r2 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0]
        new_query = f'{{"o": "n", "a": {{"o": "i", "a": [{{"o": "n", "a": {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}}}, {{"o": "n", "a": {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}}}]}}}}'
    elif meta_formula == 'up-DM':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][1]
        new_query = f'{{"o": "p", "a": [[{r3}], {{"o": "n", "a": {{"o": "i", "a": [{{"o": "n", "a": {{"o": "p", "a": [[{r1}], {{"o": "e", "a": [{e1}]}}]}}}}, {{"o": "n", "a": {{"o": "p", "a": [[{r2}], {{"o": "e", "a": [{e2}]}}]}}}}]}}}}]}}'
    else:
        new_query = None
        print('not valid name!')
    return new_query


def store_json_query_with_check(
        queries, easy_answers, hard_answers, store_fold, projection_easy, projection_hard, mode):
    for beta_structure in queries.keys():
        my_train_data = collections.defaultdict(list)
        beta_name = query_name_dict[beta_structure]
        print(f'task {beta_name} parsing')
        meta_formula_v2 = beta_query_v2[beta_name]
        query_set = queries[beta_structure]
        for i, query in enumerate(query_set):
            easy_ans = easy_answers[query]
            hard_ans = hard_answers[query]
            json_form_query = transform_json_query(query, beta_name)
            query_instance = parse_formula(meta_formula_v2)
            query_instance.additive_ground(json.loads(json_form_query))
            easy_ans_check = query_instance.deterministic_query(projection_easy)
            hard_ans_check = query_instance.deterministic_query(projection_hard) - easy_ans_check
            if easy_ans_check != easy_ans:
                print(query, easy_ans, easy_ans_check)
                raise ValueError
            if hard_ans_check != hard_ans:
                print(query,  hard_ans, hard_ans_check)
                raise ValueError
            my_train_data['query'].append(json_form_query)
            my_train_data['id'].append(i)
            if mode == 'test' or mode == 'valid':
                my_train_data['easy_answer_set'].append(easy_ans)
                my_train_data['hard_answer_set'].append(hard_ans)
            elif mode == 'train':
                my_train_data['answer_set'].append(hard_ans)
        df = pd.DataFrame(data=my_train_data)
        storation_path = os.path.join(store_fold, f"{mode}_{beta_name}.csv")
        df.to_csv(storation_path, index=False)


def store_query_with_check(queries, easy_answers, hard_answers, store_fold, projection_easy, projection_hard, mode):
    for beta_structure in queries.keys():
        my_train_data = collections.defaultdict(list)
        beta_name = query_name_dict[beta_structure]
        my_name = beta_query[beta_name]
        query_set = queries[beta_structure]
        for i, query in enumerate(query_set):
            easy_ans = easy_answers[query]
            hard_ans = hard_answers[query]
            our_form_query = transform_query(query, beta_name)
            query_instance = parse_foq_formula(our_form_query)
            easy_ans_check = query_instance.deterministic_query(projection_easy)
            hard_ans_check = query_instance.deterministic_query(projection_hard) - easy_ans_check
            if easy_ans_check != easy_ans:
                print(query, our_form_query, easy_ans, easy_ans_check)
                raise ValueError
            if hard_ans_check != hard_ans:
                print(query, our_form_query, hard_ans, hard_ans_check)
                raise ValueError
            my_train_data['query'].append(our_form_query)
            my_train_data['id'].append(i)
            if mode == 'test' or mode == 'valid':
                my_train_data['easy_answer_set'].append(easy_ans)
                my_train_data['hard_answer_set'].append(hard_ans)
            elif mode == 'train':
                my_train_data['answer_set'].append(hard_ans)
        df = pd.DataFrame(data=my_train_data)
        storation_path = os.path.join(store_fold, f"{mode}_{beta_name}.csv")
        df.to_csv(storation_path, index=False)


if __name__ == "__main__":
    data_path = 'data/FB15k-237-betae'
    rawdata_path = 'datasets_knowledge_embedding/FB15k-237'
    ent2id, rel2id, projection_train, reverse_projection_train, projection_valid, reverse_projection_valid,\
        projection_test, reverse_projection_test = load_rawdata_with_indexing(data_path, rawdata_path)
    train_queries = pickle.load(
        open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
    valid_easy_ans = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
    valid_hard_ans = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
    test_easy_ans = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))
    test_hard_ans = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
    import pandas as pd
    answer_none = collections.defaultdict(set)
    projection_none = collections.defaultdict(lambda :collections.defaultdict(set))
    store_path = 'data/FB15k-237-betae'
    '''
    store_query_with_check(train_queries, answer_none, train_answers,
                           store_path, projection_none, projection_train, mode='train')
    store_query_with_check(valid_queries, valid_easy_ans, valid_hard_ans,
                           store_path, projection_train, projection_valid, mode='valid')
    store_query_with_check(test_queries, test_easy_ans, test_hard_ans,
                           store_path, projection_valid, projection_test, mode='test')
                           '''
    store_json_query_with_check(train_queries, answer_none, train_answers,
                                store_path, projection_none, projection_train, mode='train')
    store_json_query_with_check(valid_queries, valid_easy_ans, valid_hard_ans,
                                store_path, projection_train, projection_valid, mode='valid')
    store_json_query_with_check(test_queries, test_easy_ans, test_hard_ans,
                                store_path, projection_valid, projection_test, mode='test')
