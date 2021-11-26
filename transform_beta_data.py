from collections import defaultdict
import os
from os.path import join
import pickle
import json
import logging

import pandas as pd
from tqdm import tqdm

from fol import parse_formula, beta_query_v2, beta_query_original_form
from utils.util import load_data_with_indexing
from formula_generation import count_query_depth, normal_forms_generation
from benchmark_sampling import normal_forms_transformation

query_name_dict = {
    ('e', ('r',)): '1p',
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
    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up'
    }

"""
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
"""


def transform_json_query(query, meta_formula, option=None):
    """ Prepare the dobject accordingly and then dump to json string
    Only transform into original form.
    """
    if meta_formula == '1p':
        e, r = query[0], query[1][0]
        dobject = {"o": "e", "a": [e]}
        dobject = {"o": "p", "a": [[r], dobject]}
    elif meta_formula == '2p':
        e1, r1, r2 = query[0], query[1][0], query[1][1]
        dobject = {"o": "e", "a": [e1]}
        dobject = {"o": "p", "a": [[r1], dobject]}
        dobject = {"o": "p", "a": [[r2], dobject]}
    elif meta_formula == '3p':
        e1, r1, r2, r3 = query[0], query[1][0], query[1][1], query[1][2]
        dobject = {"o": "e", "a": [e1]}
        dobject = {"o": "p", "a": [[r1], dobject]}
        dobject = {"o": "p", "a": [[r2], dobject]}
        dobject = {"o": "p", "a": [[r3], dobject]}
    elif meta_formula == '2i':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject = {"o": "i", "a": [dobject1, dobject2]}
    elif meta_formula == '3i':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject3 = {"o": "e", "a": [e3]}
        dobject3 = {"o": "p", "a": [[r3], dobject3]}
        if option == "binary":
            dobject = {"o": "i", "a": [dobject1, dobject2]}
            dobject = {"o": "i", "a": [dobject, dobject3]}
        else:
            dobject = {"o": "I", "a": [dobject1, dobject2, dobject3]}
    elif meta_formula == 'ip':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject = {"o": "i", "a": [dobject1, dobject2]}
        dobject = {"o": "p", "a": [[r3], dobject]}
    elif meta_formula == 'pi':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "p", "a": [[r2], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r3], dobject2]}
        dobject = {"o": "i", "a": [dobject2, dobject1]}
    elif meta_formula == '2in':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject2 = {"o": "n", "a": dobject2}
        dobject = {"o": "i", "a": [dobject2, dobject1]}
    elif meta_formula == '3in':
        e1, e2, e3, r1, r2, r3 = query[0][0], query[1][0], query[2][0], query[0][1][0], query[1][1][0], query[2][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject3 = {"o": "e", "a": [e3]}
        dobject3 = {"o": "p", "a": [[r3], dobject3]}
        dobject3 = {"o": "n", "a": dobject3}
        if option == "binary":
            dobject = {"o": "i", "a": [dobject1, dobject2]}
            dobject = {"o": "i", "a": [dobject, dobject3]}
        else:
            dobject = {"o": "I", "a": [dobject3, dobject1, dobject2]}
    elif meta_formula == 'inp':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject2 = {"o": "n", "a": dobject2}
        dobject = {"o": "i", "a": [dobject2, dobject1]}
        dobject = {"o": "p", "a": [[r3], dobject]}
    elif meta_formula == 'pin':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "p", "a": [[r2], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r3], dobject2]}
        dobject2 = {"o": "n", "a": dobject2}
        dobject = {"o": "i", "a": [dobject2, dobject1]}
    elif meta_formula == 'pni':
        e1, e2, r1, r2, r3 = query[0][0], query[1][0], query[0][1][0], query[0][1][1], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "p", "a": [[r2], dobject1]}
        dobject1 = {"o": "n", "a": dobject1}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r3], dobject2]}
        dobject = {"o": "i", "a": [dobject1, dobject2]}
    elif meta_formula == '2u-DNF' or meta_formula == '2u':
        e1, e2, r1, r2 = query[0][0], query[1][0], query[0][1][0], query[1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject = {"o": "u", "a": [dobject1, dobject2]}
    elif meta_formula == 'up':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject = {"o": "u", "a": [dobject1, dobject2]}
        dobject = {"o": "p", "a": [[r3], dobject]}
    elif meta_formula == 'up-DNF':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "p", "a": [[r3], dobject1]}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject2 = {"o": "p", "a": [[r3], dobject2]}
        dobject = {"o": "u", "a": [dobject1, dobject2]}
    elif meta_formula == '2u-DM':
        e1, e2, r1, r2 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "n", "a": dobject1}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject2 = {"o": "n", "a": dobject2}
        dobject = {"o": "i", "a": [dobject1, dobject2]}
        dobject = {"o": "n", "a": dobject}
    elif meta_formula == 'up-DM':
        e1, e2, r1, r2, r3 = query[0][0][0], query[0][1][0], query[0][0][1][0], query[0][1][1][0], query[1][1]
        dobject1 = {"o": "e", "a": [e1]}
        dobject1 = {"o": "p", "a": [[r1], dobject1]}
        dobject1 = {"o": "n", "a": dobject1}
        dobject2 = {"o": "e", "a": [e2]}
        dobject2 = {"o": "p", "a": [[r2], dobject2]}
        dobject2 = {"o": "n", "a": dobject2}
        dobject = {"o": "i", "a": [dobject1, dobject2]}
        dobject = {"o": "n", "a": dobject}
        dobject = {"o": "p", "a": [[r3], dobject]}
    else:
        raise NotImplementedError
    return json.dumps(dobject)


def store_json_query_with_check(
        queries, easy_answers, hard_answers,
        store_fold, projection_easy, projection_hard,
        mode, beta_names=None):
    for beta_structure in queries.keys():
        beta_name = query_name_dict[beta_structure]
        if beta_names is not None and beta_name not in beta_names:
            continue
        meta_formula_v2 = beta_query_v2[beta_name]

        logging.info(f"handling {beta_structure}, ({beta_name})"
                     f" with formula {beta_query_v2[beta_name]}")

        my_train_data = defaultdict(list)

        query_set = queries[beta_structure]

        for i, query in tqdm(enumerate(query_set)):

            easy_ans = easy_answers[query]
            hard_ans = hard_answers[query]

            json_form_query = transform_json_query(query, beta_name)
            query_instance = parse_formula(meta_formula_v2)
            query_instance.additive_ground(json.loads(json_form_query))
            easy_ans_check = query_instance.deterministic_query(
                projection_easy)
            hard_ans_check = query_instance.deterministic_query(
                projection_hard) - easy_ans_check

            if easy_ans_check != easy_ans:
                logging.error(query, easy_ans, easy_ans_check)
                raise ValueError
            if hard_ans_check != hard_ans:
                logging.error(query, hard_ans, hard_ans_check)
                raise ValueError

            my_train_data['query'].append(json_form_query)
            my_train_data['id'].append(i)

            if mode == 'test' or mode == 'valid':
                my_train_data['easy_answer_set'].append(easy_ans)
                my_train_data['hard_answer_set'].append(hard_ans)
            elif mode == 'train':
                my_train_data['answer_set'].append(hard_ans)

        df = pd.DataFrame(data=my_train_data)
        storation_path = join(store_fold, f"{mode}_{beta_name}.csv")
        logging.info(f"{len(df)} queries is obtained")
        df.to_csv(storation_path, index=False)


def store_json_query_benchmark_form_with_check(
        queries, easy_answers, hard_answers,
        store_fold, projection_easy, projection_hard,
        mode, str_length=4, beta_names=None):
    formula_id_dict = defaultdict(list)
    for structure_index, beta_structure in enumerate(queries.keys()):
        if beta_structure not in query_name_dict:
            continue
        beta_name = query_name_dict[beta_structure]
        if beta_names is not None and beta_name not in beta_names:
            continue
        meta_formula_v2 = beta_query_original_form[beta_name]
        result = normal_forms_generation(meta_formula_v2)
        for key in result:
            formula_id_dict[key].append(result[key])
        formula_id_dict['num_anchor_nodes'].append(meta_formula_v2.count('e'))
        formula_id_dict['formula_id'].append(f"type{structure_index:0{str_length}d}")

        logging.info(f"handling {beta_structure}, ({beta_name})"
                     f" with formula {beta_query_original_form[beta_name]}")
        my_train_data = defaultdict(list)
        query_set = queries[beta_structure]
        for i, query in tqdm(enumerate(query_set)):
            easy_ans, hard_ans = easy_answers[query], hard_answers[query]
            json_form_query = transform_json_query(query, beta_name, option='binary')
            query_instance = parse_formula(meta_formula_v2)
            query_instance.additive_ground(json.loads(json_form_query))
            assert query_instance.formula == meta_formula_v2
            easy_ans_check = query_instance.deterministic_query(projection_easy)
            hard_ans_check = query_instance.deterministic_query(projection_hard) - easy_ans_check
            if easy_ans_check != easy_ans:
                logging.error(query, easy_ans, easy_ans_check)
                raise ValueError
            if hard_ans_check != hard_ans:
                logging.error(query, hard_ans, hard_ans_check)
                raise ValueError
            if mode == 'test' or mode == 'valid':
                my_train_data['easy_answer_set'].append(easy_ans)
                my_train_data['hard_answer_set'].append(hard_ans)
            elif mode == 'train':
                my_train_data['answer_set'].append(hard_ans)
            else:
                assert False, 'Not valid mode!'
            results = normal_forms_transformation(query_instance)
            for normal_form in results:
                my_train_data[normal_form].append(results[normal_form].dumps)
        df = pd.DataFrame(data=my_train_data)
        storation_path = join(store_fold, f"{mode}-type{structure_index:0{str_length}d}.csv")
        logging.info(f"{len(df)} queries is obtained")
        df.to_csv(storation_path, index=False)
    df_formula = pd.DataFrame(data=formula_id_dict)
    df_formula.to_csv(join(store_fold, f"{mode}_formulas.csv"), index=False)


"""
def store_query_with_check(queries, easy_answers, hard_answers, store_fold, projection_easy, projection_hard, mode):
    for beta_structure in queries.keys():
        my_train_data = defaultdict(list)
        beta_name = query_name_dict[beta_structure]
        my_name = beta_query[beta_name]
        query_set = queries[beta_structure]
        for i, query in enumerate(query_set):
            easy_ans = easy_answers[query]
            hard_ans = hard_answers[query]
            our_form_query = transform_query(query, beta_name)
            query_instance = parse_foq_formula(our_form_query)
            easy_ans_check = query_instance.deterministic_query(
                projection_easy)
            hard_ans_check = query_instance.deterministic_query(
                projection_hard) - easy_ans_check
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
        storation_path = join(store_fold, f"{mode}_{beta_name}.csv")
        df.to_csv(storation_path, index=False)
"""

if __name__ == "__main__":
    beta_data_folders = ["data/FB15k-237-betae",
                         "data/FB15k-betae",
                         "data/NELL-betae"]
    os.makedirs(name="logs", exist_ok=True)
    logging.basicConfig(filename="logs/transform_beta_data.log",
                        filemode="wt",
                        level=logging.INFO)
    logging.info("begin transfer beta data")

    for data_path in beta_data_folders:
        logging.info(f"beta data folder {data_path}")
        target_data_path = data_path
        logging.info(f"target data folder {target_data_path}")
        ent2id, rel2id, \
        proj_train, reverse_train, \
        proj_valid, reverse_valid, \
        proj_test, reverse_test = load_data_with_indexing(data_path)

        train_queries = pickle.load(
            open(join(data_path, "train-queries.pkl"), 'rb'))
        train_ans = pickle.load(
            open(join(data_path, "train-answers.pkl"), 'rb'))

        valid_queries = pickle.load(
            open(join(data_path, "valid-queries.pkl"), 'rb'))
        valid_easy_ans = pickle.load(
            open(join(data_path, "valid-easy-answers.pkl"), 'rb'))
        valid_hard_ans = pickle.load(
            open(join(data_path, "valid-hard-answers.pkl"), 'rb'))

        test_queries = pickle.load(
            open(join(data_path, "test-queries.pkl"), 'rb'))
        test_easy_ans = pickle.load(
            open(join(data_path, "test-easy-answers.pkl"), 'rb'))
        test_hard_ans = pickle.load(
            open(join(data_path, "test-hard-answers.pkl"), 'rb'))

        answer_none = defaultdict(set)
        proj_none = defaultdict(lambda: defaultdict(set))

        store_path = target_data_path
        os.makedirs(store_path, exist_ok=True)
        '''
        store_query_with_check(train_queries, answer_none, train_answers,
                            store_path, projection_none, projection_train, mode='train')
        store_query_with_check(valid_queries, valid_easy_ans, valid_hard_ans,
                            store_path, projection_train, projection_valid, mode='valid')
        store_query_with_check(test_queries, test_easy_ans, test_hard_ans,
                            store_path, projection_valid, projection_test, mode='test')
        '''

        store_json_query_benchmark_form_with_check(train_queries,
                                                   answer_none,
                                                   train_ans,
                                                   store_path,
                                                   proj_none,
                                                   proj_train,
                                                   mode='train',
                                                   beta_names=None)
        store_json_query_benchmark_form_with_check(valid_queries,
                                                   valid_easy_ans,
                                                   valid_hard_ans,
                                                   store_path,
                                                   proj_train,
                                                   proj_valid,
                                                   mode='valid',
                                                   beta_names=None)
        store_json_query_benchmark_form_with_check(test_queries,
                                                   test_easy_ans,
                                                   test_hard_ans,
                                                   store_path,
                                                   proj_valid,
                                                   proj_test,
                                                   mode='test',
                                                   beta_names=None)
