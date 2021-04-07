import collections
import os
import pickle
from fol.sampler import *
from fol.foq import parse_foq_formula

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


if __name__ == "__main__":
    data_path = '../data/FB15k-237-betae'
    train_queries = pickle.load(
        open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(
        open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
    stanford_data_path = '../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
    projection_none = [collections.defaultdict(set) for i in range(len(all_entity_dict))]
    reverse_proection_none = [collections.defaultdict(set) for i in range(len(all_entity_dict))]
    projection_train, reverse_projection_train = load_data('../datasets_knowledge_embedding/FB15k-237/train.txt',
                                                           all_entity_dict, all_relation_dict, projection_none,
                                                           reverse_proection_none)
    import pandas as pd
    for beta_structure in query_name_dict.keys():
        my_train_data = collections.defaultdict(list)
        beta_name = query_name_dict[beta_structure]
        my_name = beta_query[beta_name]
        train_set = train_queries[beta_structure]
        for i, query in enumerate(train_set):
            ans = train_answers[query]
            our_form_query = transform_query(query, beta_name)
            query_instance = parse_foq_formula(our_form_query)
            ans_check = query_instance.deterministic_query(projection_train)
            if ans_check != ans:
                print(query, our_form_query, ans, ans_check)
                raise ValueError
            my_train_data['query'].append(our_form_query)
            my_train_data['answer_set'].append(ans)
            my_train_data['id'].append(i)
        df = pd.DataFrame(data=my_train_data)
        store_path = os.path.join('../transformed_data/FB15k-237-betae/', f"train_{beta_name}.csv")
        df.to_csv(store_path, index=False)





