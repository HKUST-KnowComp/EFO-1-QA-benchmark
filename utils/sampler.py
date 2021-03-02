"""
sample first-order logic KB queries and answers.
"""
import collections
import os
import pickle
import random
import numpy as np

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

e,r=1,3
t1=tuple([e, (r,)])
t2=(e,(r,))
ans=(t1==t2)
def produce_single_query(incoming_edges,outcoming_edges,projection,reverse_projection,nentity,query_structure):
    if(query_structure == '1p'):
        e2 = random.randint(0, nentity)
        while(len(incoming_edges[e2]) == 0):
            e2 = random.randint(0, nentity)
        r = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r)], 1)[0]
        query = (e1, (-r,))
        ans = projection[(e1, r)]
    elif(query_structure == '2p'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
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
    elif(query_structure == '2i'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)))
        ans = projection[(e1,r1)].intersection(projection[(e2,r2)])
    elif(query_structure == '3i'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r1, r2 ,r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3,)))
        ans = projection[(e1,r1)].intersection(projection[(e2, r2)]).intersection(projection[(e3, r3)])
    elif(query_structure == '2in'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2, -2)))
        ans = projection[(e1, r1)] - projection[(e2, r2)]
    elif(query_structure == '3in'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r1, r2 ,r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3, -2)))
        ans = projection[(e1,  r1)].intersection(projection[(e2, r2)]) - projection[(e3, r3)]
    elif(query_structure == 'inp'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r3 = random.sample(incoming_edges[e4], 1)[0]
        e3 = random.sample(reverse_projection[(e4, r3)], 1)[0]
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = (((e1, (-r1,)), (e2, (-r2, -2))), (-r3,))
        ans = set()
        for e_3 in projection[(e1,r1)] - projection[(e2,r2)]:
            ans.update(projection[(e_3, r3)])
    elif(query_structure == 'pin'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)])
        query = ((e1, (-r1, -r2)), (e3, (-r3, -2)))
        ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)] - projection[(e3, r3)])
    elif(query_structure == 'pni'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)])
        query = ((e1, (-r1, -r2, -2)), (e3, (-r3,)))
        ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e3, r3)] - projection[(e_2, r2)])

    return query, ans


def produce_query(incoming_edges,outcoming_edges,projection,reverse_projection,nentity,query_num,query_structures):
    all_queries = collections.defaultdict(set)
    all_answers = collections.defaultdict(set)
    for query_structure in query_structures:
        for i in range(query_num):
            query, ans= produce_single_query(incoming_edges, outcoming_edges, projection, reverse_projection, nentity,query_structure)
            all_queries[query_structure].add(query)
            all_answers[query] = ans
    return all_queries, all_answers


def produce_single_test_query(incoming_edges, outcoming_edges, projection, reverse_projection, projection_origin, reverse_projection_origin, nentity,query_structure):
    if(query_structure == '1p'):
        e2 = random.randint(0, nentity)
        while(len(incoming_edges[e2]) == 0):
            e2 = random.randint(0, nentity)
        r = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2,r)], 1)[0]
        query = (e1, (-r,))
        ans = projection[(e1, r)]
        easy_ans = projection_origin[(e1, r)]
    elif(query_structure == '2p'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
            e3 = random.randint(0, nentity)
        r2 = random.sample(incoming_edges[e3], 1)[0]
        e2 = random.sample(reverse_projection[(e3, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)], 1)[0]
        query = (e1, (-r1, -r2))
        ans = set()
        for e_2 in projection[(e1,r1)]:
            ans.update(projection[(e_2,r2)])
        easy_ans = set()
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans.update(projection_origin[(e_2, r2)])
    elif(query_structure == '2i'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)])
        easy_ans = projection_origin[(e1, r1)].intersection(projection_origin[(e2, r2)])
    elif(query_structure == '3i'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r1, r2 ,r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3,)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]).intersection(projection[(e3, r3)])
        easy_ans = projection_origin[(e1, r1)].intersection((projection_origin[(e2, r2)])).intersection((projection_origin[(e3, r3)]))
    elif(query_structure == '2in'):
        e3 = random.randint(0, nentity)
        while(len(incoming_edges[e3]) == 0):
            e3 = random.randint(0, nentity)
        r1, r2 = random.sample(incoming_edges[e3], 1)[0], random.sample(incoming_edges[e3], 1)[0]
        e1, e2 = random.sample(reverse_projection[(e3, r1)], 1)[0], random.sample(reverse_projection[(e3, r2)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2, -2)))
        ans = projection[(e1, r1)] - projection[(e2, r2)]
        easy_ans = projection_origin[(e1, r1)] - projection_origin[(e2, r2)]
    elif(query_structure == '3in'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r1, r2 ,r3 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e1, e2, e3 = random.sample(reverse_projection[(e4, r1)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0], random.sample(reverse_projection[(e4, r3)], 1)[0]
        query = ((e1, (-r1,)), (e2, (-r2,)), (e3, (-r3, -2)))
        ans = projection[(e1, r1)].intersection(projection[(e2, r2)]) - projection[(e3, r3)]
        easy_ans = projection_origin[(e1, r1)].intersection(projection_origin[(e2, r2)]) - projection_origin[(e3, r3)]
    elif(query_structure == 'inp'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
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
    elif(query_structure == 'pin'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)])
        query = ((e1, (-r1, -r2)), (e3, (-r3, -2)))
        ans = set()
        easy_ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e_2, r2)] - projection[(e3, r3)])
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans.update((projection_origin[(e2, r2)]) - projection_origin[(e3, r3)])
    elif(query_structure == 'pni'):
        e4 = random.randint(0, nentity)
        while(len(incoming_edges[e4]) == 0):
            e4 = random.randint(0, nentity)
        r3, r2 = random.sample(incoming_edges[e4], 1)[0], random.sample(incoming_edges[e4], 1)[0]
        e3, e2 = random.sample(reverse_projection[(e4, r3)], 1)[0], random.sample(reverse_projection[(e4, r2)], 1)[0]
        r1 = random.sample(incoming_edges[e2], 1)[0]
        e1 = random.sample(reverse_projection[(e2, r1)])
        query = ((e1, (-r1, -r2, -2)), (e3, (-r3,)))
        ans = set()
        easy_ans = set()
        for e_2 in projection[(e1, r1)]:
            ans.update(projection[(e3, r3)] - projection[(e_2, r2)])
        for e_2 in projection_origin[(e1, r1)]:
            easy_ans.update(projection_origin[(e3, r3)] - projection_origin[(e_2, r2)])
    hard_ans = ans - easy_ans
    return query, easy_ans, hard_ans
def produce_test_query(incoming_edges, outcoming_edges, projection,reverse_projection, projection_origin, reverse_projection_origin, nentity, query_num, query_structures):
    easy_queries = collections.defaultdict(set)
    hard_queries = collections.defaultdict(set)
    all_easy_answers = collections.defaultdict(set)
    all_hard_answers = collections.defaultdict(set)
    for query_structure in query_structures:
        for i in range(query_num):
            query, easy_ans, hard_ans= produce_single_test_query(incoming_edges, outcoming_edges, projection, reverse_projection, projection_origin, reverse_projection_origin, nentity,query_structure)
            if(len(hard_ans) == 0):
                easy_queries[query_structure].add(query)
            else:
                hard_queries[query_structure].add(query)
            all_easy_answers[query] = easy_ans
            all_hard_answers[query] = hard_ans
    return easy_queries, hard_queries, all_easy_answers, all_hard_answers

def read_indexing(data_path):
    ent2id = pickle.load(
        open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(
        open(os.path.join(data_path, "rel2id.pkl"), 'rb'))
    return ent2id, rel2id
def check_query(queries, answer1, answer2):
    for query in queries:
        if(answer1[query] != answer2[query]):
            print(query)
            return False
    return True
stanford_data_path = '../data/FB15k-237-betae'

data_path = '../datasets_knowledge_embedding/FB15k-237'
new_data_path = '../my_data/FB15k-237'
query_num = 3
all_entity_dict, all_relation_dict = read_indexing(stanford_data_path)
nentity, nrelation = int(0), int(0) #14951 -1345
incoming_edges = collections.defaultdict(set)
outcoming_edges = collections.defaultdict(set)
projection = collections.defaultdict(set)
reverse_projection = collections.defaultdict(set)
with open('../datasets_knowledge_embedding/FB15k-237/train.txt','r',errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        if e1 not in all_entity_dict:
            all_entity_dict[e1] = nentity
            nentity += 1
        if e2 not in all_entity_dict:
            all_entity_dict[e2] = nentity
            nentity += 1
        if r not in all_relation_dict:
            all_relation_dict[r] = nrelation
            nrelation -= 1
        e1, r, e2 = all_entity_dict[e1], all_relation_dict[r], all_entity_dict[e2]
        incoming_edges[e2].add(r)
        outcoming_edges[e1].add(r)
        projection[(e1, r)].add(e2)
        reverse_projection[(e2, r)].add(e1)

print('nentity,nrelation',nentity,nrelation)
my_tasks = ['1p', '2i', '2in', '3in', 'inp', 'pni', 'pin']
train_queries, train_answers = produce_query(incoming_edges,outcoming_edges,projection,reverse_projection,nentity,query_num=query_num, query_structures=my_tasks)
pickle.dump(train_queries, open(os.path.join(
    new_data_path, "train-queries.pkl"), "wb"))
pickle.dump(train_answers, open(os.path.join(
    new_data_path, "train-answers.pkl"), "wb"))
#print(len(projection[(403,-91)]))

incoming_edges_valid = incoming_edges
outcoming_edges_valid = outcoming_edges
projection_valid = projection
reverse_projection_valid = reverse_projection
with open('../datasets_knowledge_embedding/FB15k/my_valid.txt','r',errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        e1, r, e2 = all_entity_dict[e1], all_relation_dict[r], all_entity_dict[e2]
        incoming_edges_valid[e2].add(r)
        outcoming_edges_valid[e1].add(r)
        projection_valid[(e1,r)].add(e2)
        reverse_projection_valid[(e2,r)].add(e1)

valid_easy_query, valid_hard_query, valid_easy_ans, valid_hard_ans = produce_test_query(incoming_edges_valid, outcoming_edges_valid, projection_valid, reverse_projection_valid, projection, reverse_projection, nentity, query_num=query_num, query_structures=my_tasks)
pickle.dump(valid_hard_query, open(os.path.join(
    new_data_path, "valid-queries.pkl"), "wb"))
pickle.dump(valid_easy_ans, open(os.path.join(
    new_data_path, "valid-easy-answers.pkl"), "wb"))
pickle.dump(valid_hard_ans, open(os.path.join(
    new_data_path, "valid-hard-answers.pkl"), "wb"))

incoming_edges_test = incoming_edges_valid
outcoming_edges_test = outcoming_edges_valid
projection_test = projection_valid
reverse_projection_test = reverse_projection_valid
with open('../datasets_knowledge_embedding/FB15k/my_test.txt','r',errors='ignore') as infile:
    for line in infile.readlines():
        e1, r, e2 = line.strip().split('\t')
        e1, r, e2 = all_entity_dict[e1], all_relation_dict[r], all_entity_dict[e2]
        incoming_edges_test[e2].add(r)
        outcoming_edges_test[e1].add(r)
        projection_test[(e1,r)].add(e2)
        reverse_projection_test[(e2,r)].add(e1)

test_easy_query, test_hard_query, test_easy_ans, test_hard_ans = produce_test_query(incoming_edges_test, outcoming_edges_test, projection_test, reverse_projection_test, projection_valid, reverse_projection_valid, nentity, query_num=query_num, query_structures=my_tasks)
pickle.dump(test_hard_query, open(os.path.join(
    new_data_path, "test-queries.pkl"), "wb"))
pickle.dump(test_easy_ans, open(os.path.join(
    new_data_path, "test-easy-answers.pkl"), "wb"))
pickle.dump(test_hard_ans, open(os.path.join(
    new_data_path, "test-hard-answers.pkl"), "wb"))

