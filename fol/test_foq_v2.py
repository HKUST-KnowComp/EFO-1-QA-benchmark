import collections
import random
import sys
from os import path as osp
import json
import pandas as pd
from fol import beta_query_v2
from fol.foq_v2 import parse_formula
from utils.util import read_indexing, load_graph, load_data_with_indexing
sys.path.append(osp.dirname(osp.dirname(__file__)))
stanford_data_path = 'data/FB15k-237-betae'


def random_e_ground(foq_formula):
    for i, c in enumerate(foq_formula):
        if c == 'e':
            return foq_formula[:i] + "{" + str(random.randint(0, 99)) + "}" + foq_formula[i + 1:]
    raise ValueError("Nothing to gound")


def random_p_ground(foq_formula):
    for i, c in enumerate(foq_formula):
        if c == 'p':
            return foq_formula[:i] + "[" + str(random.randint(0, 99)) + "]" + foq_formula[i + 1:]
    raise ValueError("Nothing to gound")


def complete_ground(foq_formula):
    while 1:
        try:
            foq_formula = random_e_ground(foq_formula)
        except:
            break
    while 1:
        try:
            foq_formula = random_p_ground(foq_formula)
        except:
            break
    return foq_formula


def test_parse_formula():
    for k, v in beta_query_v2.items():
        obj = parse_formula(v)
        assert obj.formula == v, print(obj.formula, v)
        oobj = parse_formula(obj.formula)
        assert oobj.formula == obj.formula
        print(k, obj, obj.formula)


# we don't need this any more
def test_parse_grounded_formula():

    for k, v in beta_query_v2.items():
        gv = random_p_ground(random_e_ground(v))
        obj = parse_formula(v)
        gobj = parse_formula(gv)

        oobj = parse_formula(obj.formula)
        assert gobj.formula == oobj.formula
        '''
        ogobj = parse_formula(gobj.ground_formula)
        assert gobj.ground_formula == ogobj.ground_formula
        '''


def test_additive_ground():
    for k, v in beta_query_v2.items():
        obj = parse_formula(v)
        for _ in range(10):
            gv = random_p_ground(random_e_ground(obj.dumps))
            obj.additive_ground(json.loads(gv))
        assert obj.formula == obj.formula


'''
def test_embedding_estimation():
    for k, v in beta_query_v2.items():
        cg_formula = complete_ground(v)
        obj = parse_formula(cg_formula)
        for _ in range(10):
            cg_formula = complete_ground(v)
            obj.additive_ground(cg_formula)
        print(f"multi-instantiation for formula {obj.ground_formula}")
        obj.embedding_estimation(estimator=TransEEstimator())
'''


def test_sample():
    ent2id, rel2id, proj_train, reverse_train, proj_valid, reverse_valid, proj_test, reverse_test = \
        load_data_with_indexing(stanford_data_path)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_sample = ansclass.random_query(proj_train, cumulative=True)
        ans_check_sample = ansclass.deterministic_query(proj_train)
        assert ans_sample == ans_check_sample
        query_dumps = ansclass.dumps
        brand_new_instance = parse_formula(query_structure)
        brand_new_instance.additive_ground(json.loads(query_dumps))
        ans_another = brand_new_instance.deterministic_query(proj_train)
        assert ans_another == ans_sample
        print(ansclass.dumps)


def test_backward_sample():
    ent2id, rel2id, proj_train, reverse_train, proj_valid, reverse_valid, proj_test, reverse_test = \
        load_data_with_indexing(stanford_data_path)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_back_sample = ansclass.backward_sample(proj_train, reverse_train, requirement=None,
                                                   cumulative=True, meaningful_difference=False)
        ans_check_back_sample = ansclass.deterministic_query(proj_train)
        assert ans_check_back_sample == ans_back_sample
        query_dumps = ansclass.dumps
        check_instance = parse_formula(query_structure)
        check_instance.additive_ground(json.loads(query_dumps))
        ans_another = check_instance.deterministic_query(proj_train)
        assert ans_another == ans_check_back_sample
        print(name, ansclass.dumps)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_back_sample = ansclass.backward_sample(proj_train, reverse_train, requirement=None,
                                                   cumulative=True, meaningful_difference=True)
        ans_check_back_sample = ansclass.deterministic_query(proj_train)
        assert ans_check_back_sample == ans_back_sample
        query_dumps = ansclass.dumps
        check_instance = parse_formula(query_structure)
        check_instance.additive_ground(json.loads(query_dumps))
        ans_another = check_instance.deterministic_query(proj_train)
        assert ans_another == ans_check_back_sample
        print(name, ansclass.dumps)


def test_benchmark_backward_sample():

    ent2id, rel2id, proj_train, reverse_train, proj_valid, reverse_valid, proj_test, reverse_test = \
        load_data_with_indexing(stanford_data_path)
    formula_file = "outputs/test_generated_formula_anchor_node=3.csv"
    df = pd.read_csv(formula_file)
    for i, query_structure in enumerate(df['original']):
        ansclass = parse_formula(query_structure)
        ans_back_sample = ansclass.backward_sample(proj_train, reverse_train, requirement=None,
                                                   cumulative=True, meaningful_difference=True)
        ans_check_back_sample = ansclass.deterministic_query(proj_train)
        assert ans_check_back_sample == ans_back_sample
        query_dumps = ansclass.dumps
        check_instance = parse_formula(query_structure)
        check_instance.additive_ground(json.loads(query_dumps))
        ans_another = check_instance.deterministic_query(proj_train)
        assert ans_another == ans_check_back_sample
        print(i, ansclass.dumps)


if __name__ == "__main__":
    test_parse_formula()
    test_sample()
    test_backward_sample()
    test_benchmark_backward_sample()
    # test_additive_ground()
    # test_embedding_estimation()
    # test_parse_grounded_formula()
    # test_gen_foq_meta_formula()
