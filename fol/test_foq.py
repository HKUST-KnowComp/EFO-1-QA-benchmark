import random
from .foq import *


def random_e_ground(foq_formula):
    for i, c in enumerate(foq_formula):
        if c == 'e':
            return foq_formula[:i] + "{" + str(random.randint(0, 100)) + "}" + foq_formula[i+1:]

def random_p_ground(foq_formula):
    for i, c in enumerate(foq_formula):
        if c == 'p':
            return foq_formula[:i] + "[" + str(random.randint(100, 200)) + "]" + foq_formula[i+1:]

beta_query = {
    '1p': 'p(e)',
    '2p': 'p(p(e))',
    '3p': 'p(p(p(e)))',
    '2i': 'p(e)&p(e)',
    '3i': 'p(e)&p(e)&p(e)',
    '2in': 'p(e)-p(e)',
    '3in': 'p(e)&p(e)-p(e)',
    'inp': 'p(p(e)-p(e))',
    'pni': 'p(p(e))-p(e)',
    'ip': 'p(p(e)&p(e))',
    'pi': 'p(e)&p(p(e))',
    '2u': 'p(e)|p(e)',
    'up': 'p(p(e)|p(e))'
}


def test_parse_meta_formula():
    for k, v in beta_query.items():
        obj = parse_foq_formula(v)
        oobj = parse_foq_formula(obj.meta_formula)

        assert oobj.meta_formula == obj.meta_formula


def test_parse_grounded_formula():
    for k, v in beta_query.items():
        gv = random_p_ground(random_e_ground(v))
        obj = parse_foq_formula(v)
        gobj = parse_foq_formula(gv)

        oobj = parse_foq_formula(obj.meta_formula)
        assert gobj.meta_formula == oobj.meta_formula

        ogobj = parse_foq_formula(gobj.ground_formula)
        assert gobj.ground_formula == ogobj.ground_formula


def test_additive_ground():
    for k, v in beta_query.items():
        obj = parse_foq_formula(v)
        for _ in range(10):
            gv = random_p_ground(random_e_ground(v))
            obj.additive_ground(gv)
