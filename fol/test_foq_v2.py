import collections
import random
import sys
import json
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
from fol import  beta_query_v2
from fol.appfoq import TransEEstimator
from fol.foq_v2 import parse_formula
from utils.util import read_indexing, load_graph


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
            obj.additive_ground(gv)
        assert obj.formula == obj.formula


def test_embedding_estimation():
    for k, v in beta_query_v2.items():
        cg_formula = complete_ground(v)
        obj = parse_formula(cg_formula)
        for _ in range(10):
            cg_formula = complete_ground(v)
            obj.additive_ground(cg_formula)
        print(f"multi-instantiation for formula {obj.ground_formula}")
        obj.embedding_estimation(estimator=TransEEstimator())


def test_sample():
    stanford_data_path = '../../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(
        stanford_data_path)  # TODO: this function may be moved to other data utilities
    projection_none = {}
    reverse_proection_none = {}
    for i in all_entity_dict.values():
        projection_none[i] = collections.defaultdict(set)
        reverse_proection_none[i] = collections.defaultdict(set)
    projection_train, reverse_projection_train = load_graph('../../data/FB15k-237-betae/train.txt', projection_none,
                                                            reverse_proection_none)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_sample = ansclass.random_query(projection_train, cumulative=True)
        ans_check_sample = ansclass.deterministic_query(projection_train)
        assert ans_sample == ans_check_sample
        query_dumps = ansclass.dumps
        brand_new_instance = parse_formula(query_structure)
        brand_new_instance.additive_ground(json.loads(query_dumps))
        ans_another = brand_new_instance.deterministic_query(projection_train)
        assert ans_another == ans_sample
        print(ansclass.dumps)


def test_backward_sample():
    stanford_data_path = '../../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(
        stanford_data_path)  # TODO: this function may be moved to other data utilities
    projection_none = {}
    reverse_proection_none = {}
    for i in all_entity_dict.values():
        projection_none[i] = collections.defaultdict(set)
        reverse_proection_none[i] = collections.defaultdict(set)
    projection_train, reverse_projection_train = load_graph('../../data/FB15k-237-betae/train.txt', projection_none,
                                                            reverse_proection_none)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_back_sample = ansclass.backward_sample(projection_train, reverse_projection_train, requirement=None,
                                                   cumulative=True, meaningful_difference=False)
        ans_check_back_sample = ansclass.deterministic_query(projection_train)
        assert ans_check_back_sample == ans_back_sample
        query_dumps = ansclass.dumps
        check_instance = parse_formula(query_structure)
        check_instance.additive_ground(json.loads(query_dumps))
        ans_another = check_instance.deterministic_query(projection_train)
        assert ans_another == ans_check_back_sample
        print(name, ansclass.dumps)
    for name in beta_query_v2:
        query_structure = beta_query_v2[name]
        ansclass = parse_formula(query_structure)
        ans_back_sample = ansclass.backward_sample(projection_train, reverse_projection_train, requirement=None,
                                                   cumulative=True, meaningful_difference=True)
        ans_check_back_sample = ansclass.deterministic_query(projection_train)
        assert ans_check_back_sample == ans_back_sample
        query_dumps = ansclass.dumps
        check_instance = parse_formula(query_structure)
        check_instance.additive_ground(json.loads(query_dumps))
        ans_another = check_instance.deterministic_query(projection_train)
        assert ans_another == ans_check_back_sample
        print(name, ansclass.dumps)


def test_gen_foq_meta_formula():
    for i in range(100):
        mf = gen_foq_meta_formula()
        parse_foq_formula(mf)


if __name__ == "__main__":
    test_parse_formula()
    test_sample()
    test_backward_sample()
    # test_additive_ground()
    # test_embedding_estimation()
    # test_parse_grounded_formula()
    # test_gen_foq_meta_formula()
