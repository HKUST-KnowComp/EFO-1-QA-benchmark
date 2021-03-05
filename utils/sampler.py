"""
sample first-order logic KB queries and answers.
"""
import collections
import os
import pickle
import random
import numpy as np
import copy

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


def produce_single_query(incoming_edges, projection, reverse_projection, nentity, query_structure):
    if query_structure == '1p':
        e2 = random.randint(0, nentity)
        while len(incoming_edges[e2]) == 0:
            e2 = random.randint(0, nentity)
        r = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r)], 1)[0]
        query = (e1, (-r,))
        ans = projection[(e1, r)]
    elif query_structure == '2p':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r2 = random.sample(incoming_edges[e3], 1)[0]
        e2 = random.sample(reverse_projection[(e3, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = (e1, (-r1, -r2))
        ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)])
        ans = set()
    elif query_structure == '2i':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)])
    elif query_structure == '3i':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r1, r2, r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], \
                     random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[
            0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]).intersection(projection[(e3, r3)])
    elif query_structure == '2in':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2, -2)))
        ans = projection[(e1, r1)] - projection[(e2, r2)]
    elif query_structure == '3in':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r1, r2, r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], \
                     random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[
            0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3, -2)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]) - projection[(e3, r3)]
    elif query_structure == 'inp':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r3 = random.sample(incoming_edges[e4], 1)[0]
        e3 = random.sample(reverse_projection[(e4, r3)], 1)[0]
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = (((e1, (-r1,)), (e2, (-r2, -2))), (-r3,))
        ans = set()
        for e_3 in projection[(e1, r1)] - projection[(e2, r2)]:
            ans.update(projection[(e_3, r3)])
    elif query_structure == 'pin':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = ((e1, (-r1, -r2)), (e3, (-r3, -2)))
        ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)])
        ans = ans - projection[(e3, r3)]
    elif query_structure == 'pni':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = ((e1, (-r1, -r2, -2)), (e3, (-r3,)))
        ans = projection[(e3, r3)]
        for e_2 in projection[(e1, r1)]:
            ans = ans - projection[(e_2, r2)]

    return query, ans


def produce_query(incoming_edges, projection, reverse_projection, nentity, query_num, query_structures, max_answer):
    all_queries = collections.defaultdict(set)
    all_answers = collections.defaultdict(set)
    for query_structure in query_structures:
        while len(all_queries[query_structure]) < query_num:
            query, ans = produce_single_query(incoming_edges, projection, reverse_projection, nentity, query_structure)
            if len(ans) < max_answer:
                all_queries[query_structure].add(query)
                all_answers[query] = ans
    return all_queries, all_answers


def produce_single_test_query(incoming_edges, projection, reverse_projection, incoming_edges_origin,
                              projection_origin, nentity, query_structure):
    if query_structure == '1p':
        e2 = random.randint(0, nentity)
        while len(incoming_edges[e2]) == 0:
            e2 = random.randint(0, nentity)
        r = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r)], 1)[0]
        query = (e1, (-r,))
        ans = projection[(e1, r)]
        easy_ans = projection_origin[(e1, r)]
    elif query_structure == '2p':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r2 = random.sample(incoming_edges[e3], 1)[0]
        e2 = random.sample(reverse_projection[(e3, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = (e1, (-r1, -r2))
        ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)])
        easy_ans = set()
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans.update(projection_origin[(e_2, r2)])
    elif query_structure == '2i':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)])
        easy_ans = projection_origin[(e1, r1)].intersection(projection_origin[(e2, r2)])
    elif query_structure == '3i':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r1, r2, r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], \
                     random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[
            0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]).intersection(projection[(e3, r3)])
        easy_ans = projection_origin[(e1, r1)].intersection((projection_origin[(e2, r2)])).intersection(
            (projection_origin[(e3, r3)]))
    elif query_structure == '2in':
        e3 = random.randint(0, nentity)
        while len(incoming_edges[e3]) == 0:
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2, -2)))
        ans = projection[(e1, r1)] - projection[(e2, r2)]
        easy_ans = projection_origin[(e1, r1)] - projection_origin[(e2, r2)]
    elif query_structure == '3in':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r1, r2, r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], \
                     random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[
            0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3, -2)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]) - projection[(e3, r3)]
        easy_ans = projection_origin[(e1, r1)].intersection(projection_origin[(e2, r2)]) - projection_origin[(e3, r3)]
    elif query_structure == 'inp':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r3 = random.sample(incoming_edges[e4], 1)[0]
        e3 = random.sample(reverse_projection[(e4, r3)], 1)[0]
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = (((e1, (-r1,)), (e2, (-r2, -2))), (-r3,))
        ans = set()
        easy_ans = set()
        for e_3 in projection[(e1, r1)] - projection[(e2, r2)]:
            ans.update(projection[(e_3, r3)])
        for e_3 in projection_origin[(e1, r1)] - projection_origin[(e2, r2)]:
            easy_ans.update(projection_origin[(e_3, r3)])
    elif query_structure == 'pin':
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = ((e1, (-r1, -r2)), (e3, (-r3, -2)))
        ans = set()
        easy_ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)])
        ans = ans - projection[(e3, r3)]
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans.update((projection_origin[(e_2, r2)]))
        easy_ans = easy_ans - projection_origin[(e3, r3)]
    elif query_structure == 'pni':  # 自己强行分类讨论了
        e4 = random.randint(0, nentity)
        while len(incoming_edges[e4]) == 0:
            e4 = random.randint(0, nentity)
        new_edges = incoming_edges[e4] # - incoming_edges_origin[e4]
        r3, r2 = random.sample(new_edges, 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = ((e1, (-r1, -r2, -2)), (e3, (-r3,)))
        ans = projection[(e3, r3)]
        easy_ans = projection_origin[(e3, r3)]
        for e_2 in projection[(e1, r1)]:
            ans = ans - projection[(e_2, r2)]
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans = easy_ans - projection_origin[(e_2, r2)]
    hard_ans = ans - easy_ans
    return query, easy_ans, hard_ans


def produce_test_query(incoming_edges, projection, reverse_projection, incoming_edges_origin, projection_origin, nentity, query_num,
                       query_structures, max_ans):
    easy_queries = collections.defaultdict(set)
    hard_queries = collections.defaultdict(set)
    all_easy_answers = collections.defaultdict(set)
    all_hard_answers = collections.defaultdict(set)
    maximum = 10 * query_num
    for query_structure in query_structures:
        all_num = 0
        while len(hard_queries[query_structure]) < query_num and all_num < maximum:
            query, easy_ans, hard_ans = produce_single_test_query(incoming_edges, projection, reverse_projection, incoming_edges_origin,
                                                                  projection_origin, nentity, query_structure)
            if len(easy_ans) + len(hard_ans) < max_ans:
                if len(hard_ans) == 0:
                    easy_queries[query_structure].add(query)
                else:
                    hard_queries[query_structure].add(query)
                all_easy_answers[query] = easy_ans
                all_hard_answers[query] = hard_ans
                all_num += 1
    return easy_queries, hard_queries, all_easy_answers, all_hard_answers


def read_indexing(data_path):
    ent2id = pickle.load(
        open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(
        open(os.path.join(data_path, "rel2id.pkl"), 'rb'))
    id2ent = pickle.load(
        open(os.path.join(data_path, "id2ent.pkl"), 'rb'))
    id2rel = pickle.load(
        open(os.path.join(data_path, "id2rel.pkl"), 'rb'))
    return ent2id, rel2id, id2ent, id2rel


def check_query(queries, answer1, answer2):
    for q_r in queries:
        for query in queries[q_r]:
            if query in answer2.keys():
                if answer1[query] != answer2[query]:
                    print("error happen!", q_r, query, answer1[query], answer2[query])
                    return False
    return True


def count_answer_num(queries, easy_ans, hard_ans):
    easy_ans_average = collections.defaultdict(float)
    hard_ans_average = collections.defaultdict(float)
    for q_r in queries:
        for query in queries[q_r]:
            easy_ans_average[q_r] += len(easy_ans[query])
            hard_ans_average[q_r] += len(hard_ans[query])
        if len(queries[q_r]) != 0:
            easy_ans_average[q_r] /= len(queries[q_r])
            hard_ans_average[q_r] /= len(queries[q_r])
    return easy_ans_average, hard_ans_average


stanford_data_path = '../data/FB15k-237-betae'  # + 原来的， - 反向的
data_path = '../datasets_knowledge_embedding/FB15k-237'
new_data_path = '../my_data/FB15k-237'
train_query_num, valid_query_num = 10000, 5000
max_ans = 100
all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
a, b, c, d = (id2ent[3986], id2ent[7154], id2rel[132], id2rel[336])
e = id2ent[0]
numentity, numrelation = len(all_entity_dict), len(all_relation_dict)  # 14505, 237
incoming_edges = collections.defaultdict(set)
outcoming_edges = collections.defaultdict(set)
projection = collections.defaultdict(set)
reverse_projection = collections.defaultdict(set)
'''with open('../datasets_knowledge_embedding/FB15k/valid.txt', 'r', errors='ignore') as infile:
    a=1'''

with open('../datasets_knowledge_embedding/FB15k-237/train.txt', 'r', errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        r_projection = '+' + r
        r_reverse = '-' + r
        if e1 not in all_entity_dict:
            all_entity_dict[e1] = numentity
            numentity += 1
        if e2 not in all_entity_dict:
            all_entity_dict[e2] = numentity
            numentity += 1
        if r_projection not in all_relation_dict:
            all_relation_dict[r_projection] = numrelation
            numrelation -= 1
        if r_reverse not in all_relation_dict:
            all_relation_dict[r_reverse] = numrelation
            numrelation -= 1
        e1, r_projection, r_reverse, e2 = all_entity_dict[e1], -all_relation_dict[r_projection], \
                                          -all_relation_dict[r_reverse], all_entity_dict[e2]
        incoming_edges[e2].add(r_projection)
        outcoming_edges[e2].add(r_reverse)
        incoming_edges[e1].add(r_reverse)
        outcoming_edges[e1].add(r_projection)
        projection[(e1, r_projection)].add(e2)
        projection[(e2, r_reverse)].add(e1)
        reverse_projection[(e1, r_reverse)].add(e2)
        reverse_projection[(e2, r_projection)].add(e1)

my_tasks = ['1p', '2i', '2in', '3in', 'inp', 'pni', 'pin']
train_queries, train_answers = produce_query(incoming_edges, projection, reverse_projection, numentity,
                                             query_num=train_query_num, query_structures=my_tasks, max_answer=max_ans)
pickle.dump(train_queries, open(os.path.join(
    new_data_path, "train-queries.pkl"), "wb"))
pickle.dump(train_answers, open(os.path.join(
    new_data_path, "train-answers.pkl"), "wb"))

incoming_edges_valid = copy.deepcopy(incoming_edges)
outcoming_edges_valid = copy.deepcopy(outcoming_edges)
projection_valid = copy.deepcopy(projection)
reverse_projection_valid = copy.deepcopy(reverse_projection)
with open('../datasets_knowledge_embedding/FB15k-237/valid.txt', 'r', errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        r_projection = '+' + r
        r_reverse = '-' + r
        if e1 not in all_entity_dict or e2 not in all_entity_dict or r_projection not in all_relation_dict:
            pass
        else:
            e1, r_projection, r_reverse, e2 = all_entity_dict[e1], -all_relation_dict[r_projection], \
                                              -all_relation_dict[r_reverse], all_entity_dict[e2]
            incoming_edges_valid[e2].add(r_projection)
            outcoming_edges_valid[e2].add(r_reverse)
            incoming_edges_valid[e1].add(r_reverse)
            outcoming_edges_valid[e1].add(r_projection)
            projection_valid[(e1, r_projection)].add(e2)
            projection_valid[(e2, r_reverse)].add(e1)
            reverse_projection_valid[(e1, r_reverse)].add(e2)
            reverse_projection_valid[(e2, r_projection)].add(e1)

valid_easy_query, valid_hard_query, valid_easy_ans, valid_hard_ans = produce_test_query(
    incoming_edges_valid, projection_valid, reverse_projection_valid, incoming_edges, projection, numentity,
    query_num=valid_query_num, query_structures=my_tasks, max_ans=max_ans)
pickle.dump(valid_hard_query, open(os.path.join(
    new_data_path, "valid-queries.pkl"), "wb"))
pickle.dump(valid_easy_ans, open(os.path.join(
    new_data_path, "valid-easy-answers.pkl"), "wb"))
pickle.dump(valid_hard_ans, open(os.path.join(
    new_data_path, "valid-hard-answers.pkl"), "wb"))

incoming_edges_test = copy.deepcopy(incoming_edges_valid)
outcoming_edges_test = copy.deepcopy(outcoming_edges_valid)
projection_test = copy.deepcopy(projection_valid)
reverse_projection_test = copy.deepcopy(reverse_projection_valid)
with open('../datasets_knowledge_embedding/FB15k-237/test.txt', 'r', errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        r_projection = '+' + r
        r_reverse = '-' + r
        if e1 not in all_entity_dict or e2 not in all_entity_dict or r_projection not in all_relation_dict:
            pass
        else:
            e1, r_projection, r_reverse, e2 = all_entity_dict[e1], -all_relation_dict[r_projection], -all_relation_dict[
                r_reverse], all_entity_dict[e2]
            incoming_edges_test[e2].add(r_projection)
            outcoming_edges_test[e2].add(r_reverse)
            incoming_edges_test[e1].add(r_reverse)
            outcoming_edges_test[e1].add(r_projection)
            projection_test[(e1, r_projection)].add(e2)
            projection_test[(e2, r_reverse)].add(e1)
            reverse_projection_test[(e1, r_reverse)].add(e2)
            reverse_projection_test[(e2, r_projection)].add(e1)

test_easy_query, test_hard_query, test_easy_ans, test_hard_ans = produce_test_query(
    incoming_edges_test, projection_test, reverse_projection_test, incoming_edges_valid,
    projection_valid, numentity, query_num=valid_query_num, query_structures=my_tasks, max_ans=max_ans)

stanford_train_ans = pickle.load(
    open(os.path.join(stanford_data_path, "train-answers.pkl"), 'rb'))
stanford_valid_easy_ans = pickle.load(
    open(os.path.join(stanford_data_path, "valid-easy-answers.pkl"), 'rb'))
stanford_valid_hard_ans = pickle.load(
    open(os.path.join(stanford_data_path, "valid-hard-answers.pkl"), 'rb'))
stanford_test_easy_ans = pickle.load(
    open(os.path.join(stanford_data_path, "test-easy-answers.pkl"), 'rb'))
stanford_test_hard_ans = pickle.load(
    open(os.path.join(stanford_data_path, "test-hard-answers.pkl"), 'rb'))
check_ans = check_query(train_queries, train_answers, stanford_train_ans)
check_valid_ans = check_query(valid_hard_query, valid_easy_ans, stanford_valid_easy_ans) \
                  + check_query(valid_hard_query, valid_hard_ans, stanford_valid_hard_ans)
check_test_ans = check_query(test_hard_query, test_easy_ans, stanford_test_easy_ans) \
                 + check_query(test_hard_query, test_hard_ans, stanford_test_hard_ans)
print("check ans here", check_ans, check_valid_ans, check_test_ans)
easy_train_average, _ = count_answer_num(train_queries, train_answers, collections.defaultdict(set))
easy_valid_average, hard_valid_average = count_answer_num(valid_hard_query, valid_easy_ans, valid_hard_ans)
easy_test_average, hard_test_average = count_answer_num(test_hard_query, test_easy_ans, test_hard_ans)
print(easy_train_average, easy_valid_average, hard_valid_average, easy_test_average, hard_test_average)
pickle.dump(test_hard_query, open(os.path.join(
    new_data_path, "test-queries.pkl"), "wb"))
pickle.dump(test_easy_ans, open(os.path.join(
    new_data_path, "test-easy-answers.pkl"), "wb"))
pickle.dump(test_hard_ans, open(os.path.join(
    new_data_path, "test-hard-answers.pkl"), "wb"))
