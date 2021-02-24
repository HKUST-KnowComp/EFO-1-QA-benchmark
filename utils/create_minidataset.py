import collections
import os
import pickle

import numpy as np


def transform_query(query, old_new_dict, old_new_relation_dict, name):
    if name == '1p':
        # print(query,type(query),type(query[0]),type(query[1]),query[1][0],type(query[1][0]))
        e, r = old_new_dict[query[0]], old_new_relation_dict[query[1][0]]
        new_query = tuple([e, (r,)])
    elif name == '2p':
        e1, r1, r2 = old_new_dict[query[0]], query[1][0], query[1][1]
        new_query = (e1, (r1, r2))
    elif name == '3p':
        e1, r1, r2, r3 = old_new_dict[query[0]
                                      ], query[1][0], query[1][1], query[1][2]
        new_query = (e1, (r1, r2, r3))
    elif name == '2i':
        e1, e2, r1, r2 = old_new_dict[query[0][0]], old_new_dict[query[1][0]
                                                                 ], old_new_relation_dict[query[0][1][0]], old_new_relation_dict[query[1][1][0]]
        new_query = (tuple([e1, (r1,)]), tuple([e2, (r2,)]))
    elif name == '3i':
        e1, e2, e3, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], old_new_dict[query[2][0]], \
            query[0][1], \
            query[1][1], query[2][1]
        new_query = ((e1, r1), (e2, r2), (e3, r3))
    elif name == 'ip':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][1], \
            query[1]
        new_query = (((e1, r1), (e2, r2)), r3)
    elif name == 'pi':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]
                                          ], old_new_dict[1][0], query[0][1][0], query[0][1][1], query[1][1]
        new_query = ((e1, (r1, r2)), (e2, r2))
    elif name == '2in':
        e1, e2, r1, r2 = old_new_dict[query[0][0]
                                      ], old_new_dict[query[1][0]], query[0][1], query[1][1][0]
        new_query = ((e1, r1), (e2, (r2, 'n')))
    elif name == '3in':
        e1, e2, e3, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], old_new_dict[query[2][0]], \
            query[0][1], query[1][1], query[2][1][0]
        new_query = ((e1, r1), (e2, r2), (e3, (r3, 'n')))
    elif name == 'inp':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][0][
            0], query[2]
        new_query = (((e1, r1), (e2, (r2, 'n'))), r3)
    elif name == 'pin':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], query[0][1][0], query[0][1][1], \
            query[1][1][0]
        new_query = ((e1, (r1, r2)), (e2, (r3, 'n')))
    elif name == 'pni':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0]], old_new_dict[query[1][0]], query[0][1][0], query[0][1][1], \
            query[1][1]
        new_query = ((e1, (r1, r2, 'n')), (e2, r3))
    elif name == '2u-DNF':
        e1, e2, r1, r2 = old_new_dict[query[0][0]
                                      ], old_new_dict[query[1][0]], query[0][1], query[1][1]
        new_query = ((e1, r1), (e2, r2), ('u',))
    elif name == 'up-DNF':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1], query[0][1][1], \
            query[1]
        new_query = (((e1, r1), (e2, r2), ('u',)), r3)
    elif name == '2u-DM':
        e1, e2, r1, r2 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1][0], query[0][1][1][
            0]
        new_query = (((e1, (r1, 'n')), (e2, (r2, 'n'))), ('n',))
    elif name == 'up-DM':
        e1, e2, r1, r2, r3 = old_new_dict[query[0][0][0]], old_new_dict[query[0][1][0]], query[0][0][1][0], \
            query[0][1][1][0], query[1][1]
        new_query = (((e1, (r1, 'n')), (e2, (r2, 'n'))), ('n', r3))
    else:
        new_query = None
        print('not valid name!')
    return new_query


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
name_query_dict = {value: key for key, value in query_name_dict.items()}
# ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']
all_tasks = list(name_query_dict.keys())
evaluate_union = 'DNF'
data_path = 'data/FB15k-237-betae'
train_queries = pickle.load(
    open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
train_answers = pickle.load(
    open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
#valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
#valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
#test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
#test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))
nentity = 14505
number_of_queries = 100  # 10000queries  13568  1000:7331
nrelations = 474

entity_presense = np.zeros(nentity, dtype=int)
relation_presense = np.zeros(nrelations, dtype=int)
for name in all_tasks:
    query_structure = name_query_dict[name]
    train_list = list(train_queries[query_structure])[:number_of_queries]
    for i in range(number_of_queries):
        if name == '1p':
            # print(train_list[i])
            entity_presense[train_list[i][0]] = 1
            relation_presense[train_list[i][1]] = 1
            for j in train_answers[train_list[i]]:
                entity_presense[j] = 1
        elif name == '2i':
            # print(train_list[i])
            entity_presense[train_list[i][0][0]] = 1
            entity_presense[train_list[i][1][0]] = 1
            relation_presense[train_list[i][0][1]] = 1
            relation_presense[train_list[i][1][1]] = 1
            for j in train_answers[train_list[i]]:
                entity_presense[j] = 1

old_new_dict = {}
now = int(0)
for i in range(nentity):
    if entity_presense[i] == 1:
        old_new_dict[i] = now
        now += 1
print(now)
new_old_dict = {value: key for key, value in old_new_dict.items()}
old_new_relation_dict = {}
now2 = int(0)
for i in range(nrelations):
    if relation_presense[i] == 1:
        old_new_relation_dict[i] = now2
        now2 += 1
print(now2)
my_tasks = ['1p', '2i']
new_train_answers = collections.defaultdict(set)
for name in all_tasks:
    query_structure = name_query_dict[name]
    if(name in my_tasks):
        train_list = list(train_queries[query_structure])[:number_of_queries]
        new_train_list, new_valid_list, new_test_list = [], [], []
        for i in range(number_of_queries):
            train_query = train_list[i]
            train_answer = train_answers[train_query]
            new_train_query = transform_query(
                train_query, old_new_dict, old_new_relation_dict, name)
            new_train_answers[new_train_query] = set(
                [old_new_dict[answer] for answer in train_answer])
            #new_valid_query=transform_query(valid_query, old_new_dict,name)
            #new_test_query=transform_query(test_query, old_new_dict, name)
            # print(new_train_query)
            new_train_list.append(new_train_query)
            # new_valid_list.update(new_valid_query)
            # new_test_list.update(new_test_query)
        train_queries[query_structure] = set(new_train_list)
    else:  # del others
        del train_queries[query_structure]

new_data_path = 'data/FB15k-237-betae-tiny'
# print(train_queries)
# print(new_train_answers)
pickle.dump(train_queries, open(os.path.join(
    new_data_path, "train-queries.pkl"), "wb"))
pickle.dump(new_train_answers, open(os.path.join(
    new_data_path, "train-answers.pkl"), "wb"))
with open(os.path.join(new_data_path, "stats.txt"), 'w') as outfile:
    outfile.write("numentity:"+'\t'+str(now)+'\n')
    outfile.write("numrelations:"+'\t'+str(now2)+'\n')
