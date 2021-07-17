import json
import random
from abc import ABC, abstractmethod, abstractproperty
from pickle import APPEND
from typing import List, Tuple, TypedDict
from typing import Union as TUnion
from utils.dataset import Subset

import torch

from fol.appfoq import AppFOQEstimator, IntList

"""
First Order Set Query (FOSQ) with Json implementation
Each query `q` is associated to a dict object `d` (Dobject in typing system)
so that it can be serialized into a json string `s`.

Since we consider the first order logic with single free variable, then we can
construct our grammar with the set operations

We begin with the structure of the Dobject

Each Dobject contains two kv pairs:
    - 'o' for operation
        - 'e' for entity,       1 argument: list of entity_ids
        - 'p' for projection,   2 arguments: list of relation_ids, dict
        - 'd' for difference,   2 arguments: dict, dict
        - 'i' for intersection, multiple arguments: dict
        - 'u' for union,        multiple arguments: dict
    - 'a' for argument,
        which can be either a Dobject or a list of integers

d = {'o': 's',
     'a': [
         d1,
         d2,
         ...
        ]}

We use the json dumps to convert such a dict to a string.
When we refer to a string, is always "grounded", i.e. all 'e' and 'p' typed
Dobjects contains corresponding relation_id and entity_id args.

A formula by considering a lisp-like language
    (ops,arg1,arg2,...), where args = (...),
where we use () and corresponding objects to formulate a formula.
A formula is always meta, that is, we don't care about its instantiation.
A formula is the minimal information that we are used to specify the query.
Specifically, the args in one () level are *sorted by alphabetical order*.
In this way, the formula is the unique representation for a type of query.
The advantage of this grammar is that we don't need any parser.
One can use python eval() and then consider nested tuples.
"""

# Dobject is only used in a type hints, the Dobject should be used as a dict
Dobject = TypedDict('Dobject', {'o': str, 'a': List[TUnion['Dobject', int]]})
Formula = TUnion[Tuple[str, ...], Tuple['Formula', ...]]


class FirstOrderSetQuery(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def formula(self) -> str:
        """
        Where we store the structure information
        """
        pass

    @abstractmethod
    def parse(self, formula: Formula):
        """ Parse the formula
        """
        pass

    @property
    @abstractmethod
    def dumps(self) -> str:
        """
        Where we serialize the data information
        """
        pass

    @abstractmethod
    def additive_ground(self, dobject: Dobject):
        """
        Just add the values when the structure is fixed!
        """
        pass

    @abstractmethod
    def backward_sample(self, projs, rprojs,
                        contain: bool = None,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        pass

    @abstractmethod
    def deterministic_query(self, projection):
        #     Consider the first entity / relation
        pass

    @abstractmethod
    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices):
        pass

    @abstractmethod
    def random_query(self, projs, cumulative: bool = False):
        pass

    @abstractmethod
    def lift(self):
        """ Remove all intermediate objects, grounded entities and relations
        """
        pass

    @abstractmethod
    def check_ground(self) -> int:
        pass

    def __len__(self):
        return self.check_ground()

    @abstractmethod
    def to(self, device):
        pass


class Entity(FirstOrderSetQuery):
    """ A singleton set of entities
    """
    __o__ = 'e'

    def __init__(self):
        super().__init__()
        self.entities = []
        self.tentities = None
        self.device = 'cpu'

    @property
    def formula(self):
        return "(e)"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': self.entities
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, entity_list = dobject['o'], dobject['a']
        assert obj == self.__o__
        assert all(isinstance(i, int) for i in entity_list)
        self.entities.extend(entity_list)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        if self.tentities is None:
            self.tentities = torch.tensor(self.entities).to(self.device)
        if batch_indices:
            ent = self.tentities[torch.tensor(batch_indices)]
        else:
            ent = self.tentities
        return estimator.get_entity_embedding(ent)

    def lift(self):
        self.entities = []
        self.tentities = None
        return super().lift()

    def deterministic_query(self, args, **kwargs):
        # TODO: change to return a list of set
        return {self.entities[0]}

    def backward_sample(self, projs, rprojs,
                        contain: bool = None,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        if keypoint:
            if contain:
                new_entity = [keypoint]
            else:
                new_entity = random.sample(set(projs.keys()) - {keypoint}, 1)
        else:
            new_entity = random.sample(set(projs.keys()), 1)

        if cumulative:
            self.entities.append(new_entity[0])
        else:
            self.entities = new_entity

        return set(new_entity)

    def random_query(self, projs, cumulative=False):
        new_variable = random.sample(set(projs.keys()), 1)[0]
        if cumulative:
            self.entities.append(new_variable)
        else:
            self.entities = [new_variable]
        return {new_variable}

    def check_ground(self):
        return len(self.entities)

    def to(self, device):
        self.device = device
        if self.tentities is None:
            self.tentities = torch.tensor(self.entities).to(device)
        print(f'move variable object in {id(self)} to device {device}')


class Projection(FirstOrderSetQuery):
    """
    `self.relations` describes the relation ids by the KG
    """
    __o__ = 'p'

    def __init__(self, q: FirstOrderSetQuery = None):
        super().__init__()
        self.operand_q = q
        self.relations = []
        self.trelations = None
        self.device = 'cpu'

    @property
    def formula(self):
        return f"(p,{self.operand_q.meta_formula})"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': [self.relations, self.operand_q.dumps]
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, (relation_list, sub_dobject) = dobject['o'], dobject['a']
        assert obj == self.__o__
        assert all(isinstance(i, int) for i in relation_list)
        self.relations.extend(relation_list)
        self.operand_q.additive_ground(sub_dobject)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        if self.trelations is None:
            self.trelations = torch.tensor(self.relations).to(self.device)
        if batch_indices:
            rel = self.trelations[torch.tensor(batch_indices)]
        else:
            rel = self.trelations
        operand_emb = self.operand_q.embedding_estimation(estimator,
                                                          batch_indices)
        return estimator.get_projection_embedding(rel,
                                                  operand_emb)

    def lift(self):
        self.relations = []
        self.trelations = None
        return super().lift()

    def deterministic_query(self, projs):
        rel = self.relations[0]
        result = self.operand_q.deterministic_query(projs)
        answer = set()
        for e in result:
            answer.update(projs[e][rel])
        return answer

    def backward_sample(self, projs, rprojs,
                        contain: bool = None,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        # since the projection[next_point][self.rel] may contains essential_point even if not starting from it
        while True:
            if keypoint is not None:
                if contain:
                    cursor = keypoint
                else:
                    cursor = random.sample(
                        set(rprojs.keys()) - {keypoint}, 1)[0]
            else:
                cursor = random.sample(set(rprojs.keys()), 1)[0]

            relation = random.sample(rprojs[cursor].keys(), 1)[0]
            parents = rprojs[cursor][relation]
            # find an incoming edge and a corresponding node
            parent = random.sample(parents, 1)[0]

            if keypoint:
                if (keypoint in projs[parent][relation]) == contain:
                    break
            else:
                break

        p_object = self.operand_q.backward_sample(projs, rprojs,
                                                  contain=True, keypoint=parent, cumulative=cumulative, **kwargs)
        if None in p_object:  # FIXME: why this is a none in return type
            raise ValueError

        objects = set()
        for entity in p_object:  # FIXME: there used to be a copy
            if isinstance(entity, int):
                objects.update(projs[entity][relation])

        if cumulative:
            self.relations.append(relation)
        else:
            self.relations = [relation]

        return objects

    def random_query(self, projs, cumulative=False):
        variable = self.operand_q.random_query(projs, cumulative)
        objects = set()
        if len(variable) == 0:
            if cumulative:
                self.relations.append(0)
            else:
                self.relations = []  # TODO: perhaps another way to deal with it
            return objects

        chosen_variable = random.sample(variable, 1)[0]
        relation = random.sample(projs[chosen_variable].keys(), 1)[0]

        if cumulative:
            self.relations.append(relation)
        else:
            self.relations = [relation]

        objects = set()
        for e in list(variable):
            objects.update(projs[e][relation])
        return objects

    def check_ground(self):
        n_inst = self.operand_q.check_ground()
        assert len(self.relations) == n_inst
        return n_inst

    def to(self, device):
        self.device = device
        if self.trelations is None:
            self.trelations = torch.tensor(self.relations).to(device)
        print(f'move projection object in {id(self)} to device {device}')
        self.operand_q.to(device)


class MultipleSetQuery(FirstOrderSetQuery):
    def __init__(self, queries: List[FirstOrderSetQuery]):
        self.sub_queries = queries

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': [
                subq.dumps for subq in self.sub_queries
            ]
        }
        return json.dumps(dobject)

    @property
    def formula(self):
        return "({},{})".format(
            self.__o__,
            ",".join(subq.formula for subq in self.sub_queries)
        )

    def additive_ground(self, dobject: Dobject):
        obj, sub_dobjects = dobject['o'], dobject['a']
        assert obj == self.__o__
        assert len(self.sub_queries) == len(sub_dobjects)
        for q, dobj in zip(self.sub_queries, sub_dobjects):
            q.additive_ground(dobj)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        return [q.embedding_estimation(estimator, batch_indices)
                for q in self.sub_queries]

    def lift(self):
        self.loperand_q.lift()
        self.roperand_q.lift()
        return super().lift()

    def top_down_parse(self, lroperand_strs, **kwargs):
        assert len(lroperand_strs) == 2
        loperand_str, roperand_str = lroperand_strs
        lobj, largs = parse_top_foq_formula(foq_formula=loperand_str, **kwargs)
        robj, rargs = parse_top_foq_formula(foq_formula=roperand_str, **kwargs)
        self.loperand_q = lobj
        self.loperand_q.top_down_parse(largs, **kwargs)
        self.roperand_q = robj
        self.roperand_q.top_down_parse(rargs, **kwargs)

    def check_ground(self):
        l_n_inst = self.loperand_q.check_ground()
        r_n_inst = self.roperand_q.check_ground()
        assert l_n_inst == r_n_inst
        return l_n_inst

    def to(self, device):
        self.loperand_q.to(device)
        self.roperand_q.to(device)


class Intersection(MultipleSetQuery):
    __o__ = 'i'

    def __init__(self, queries: List[FirstOrderSetQuery]):
        super().__init__(queries)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        embed_list = super().embedding_estimation(estimator, batch_indices)
        # TODO: fix conjunction implementation
        return estimator.get_conjunction_embedding(embed_list)

    def deterministic_query(self, projs):
        return set.intersection(
            q.deterministic_query(projs) for q in self.sub_queries
        )

    def backward_sample(self, projs, rprojs,
                        contain: bool = True,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        if keypoint:
            if contain:
                lobjs = self.loperand_q.backward_sample(
                    projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
                robjs = self.roperand_q.backward_sample(
                    projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
            else:
                choose_formula = random.randint(0, 1)
                if choose_formula == 0:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                else:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
        else:
            keypoint = random.sample(set(projs.keys()), 1)[0]
            lobjs = self.loperand_q.backward_sample(
                projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
            robjs = self.roperand_q.backward_sample(
                projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)

        return lobjs.intersection(robjs)

    def random_query(self, projs, cumulative=False):
        lobjs = self.loperand_q.random_query(projs, cumulative)
        robjs = self.roperand_q.random_query(projs, cumulative)
        return lobjs.intersection(robjs)


class Union(MultipleSetQuery):
    __o__ = 'u'

    def __init__(self, queries: List[FirstOrderSetQuery]):
        super().__init__(queries)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        embed_list = super().embedding_estimation(estimator, batch_indices)
        # TODO: fix disjunction embeddings
        return estimator.get_disjunction_embedding(embed_list)

    def deterministic_query(self, projs):
        return set.union(
            q.deterministic_query(projs) for q in self.sub_queries
        )

    def backward_sample(self, projs, rprojs,
                        contain: bool = True,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        if keypoint:
            if contain:
                choose_formula = random.randint(0, 1)
                if choose_formula == 0:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                else:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
            else:
                lobjs = self.loperand_q.backward_sample(
                    projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
                robjs = self.roperand_q.backward_sample(
                    projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
        else:
            lobjs = self.loperand_q.backward_sample(
                projs, rprojs, cumulative=cumulative)
            robjs = self.roperand_q.backward_sample(
                projs, rprojs, cumulative=cumulative)

        return lobjs.union(robjs)

    def random_query(self, projs, cumulative=False):
        lobjs = self.loperand_q.random_query(projs, cumulative)
        robjs = self.roperand_q.random_query(projs, cumulative)
        return lobjs.union(robjs)


class Difference(FirstOrderSetQuery):
    __o__ = 'd'

    def __init__(self, lq: FirstOrderSetQuery, rq: FirstOrderSetQuery):
        self.lquery = lq
        self.rquery = rq

    @property
    def formula(self):
        return f"(d,{self.lquery.formula},{self.rquery.formula})"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': [self.lquery.dumps, self.rquery.dumps]
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, (ldobj, rdobj) = dobject['o'], dobject['a']
        assert obj == self.__o__
        self.lquery.additive_ground(ldobj)
        self.rquery.additive_ground(rdobj)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None):
        lemb, remb = super().embedding_estimation(estimator, batch_indices)
        return estimator.get_difference_embedding(lemb, remb)

    def deterministic_query(self, projs):
        l_result = self.lquery.deterministic_query(projs)
        r_result = self.lquery.deterministic_query(projs)
        return l_result - r_result

    def backward_sample(self, projs, rprojs,
                        contain: bool = True,
                        keypoint: int = None,
                        cumulative: bool = False, **kwargs):
        if keypoint:
            if contain:
                lobjs = self.loperand_q.backward_sample(
                    projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
                robjs = self.roperand_q.backward_sample(
                    projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
            else:
                choose_formula = random.randint(0, 1)
                if choose_formula == 0:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, contain=False, keypoint=keypoint, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                else:
                    lobjs = self.loperand_q.backward_sample(
                        projs, rprojs, cumulative=cumulative)
                    robjs = self.roperand_q.backward_sample(
                        projs, rprojs, contain=True, keypoint=keypoint, cumulative=cumulative)
        else:
            lobjs = self.loperand_q.backward_sample(
                projs, rprojs, cumulative=cumulative)
            robjs = self.roperand_q.backward_sample(
                projs, rprojs, cumulative=cumulative)

        return lobjs - robjs

    def random_query(self, projs, cumulative=False):
        lobjs = self.loperand_q.random_query(projs, cumulative)
        robjs = self.roperand_q.random_query(projs, cumulative)
        return lobjs - robjs


ops_dict = {
    'e': Entity,
    'p': Projection,
    'd': Difference,
    'i': Intersection,
    'u': Union
}


def parse_top_foq_formula(foq_formula: str,
                          z_obj: FirstOrderQuery = VariableQ,
                          f_obj: FirstOrderQuery = ProjectionQ,
                          binary_ops=binary_ops) -> Tuple[FirstOrderQuery, Tuple[str]]:
    """ A new function to parse top-level first-order query string
    A first-order string must:
        1. follow the meta grammar
        2. e and p are placeholders for entity (zero order object) and projection (first order object)
        3. e and p can be represented by {eid,} or [pid,] by instantiation
    The output is to breakdown a foq_formula into operations and its argument foq str into formulations like
    (class_object, [argfoqstr1, argfoqstr2])
    """

    # binary operation decision: identify the top-level binary operator and two arguments
    level_stack = []
    top_binary_ops = []
    for i, c in enumerate(foq_formula):
        if c in "()":  # if there is any composition
            if c == '(':
                level_stack.append(i)
            else:
                if level_stack:
                    begin = level_stack.pop(-1)
                    if level_stack:
                        continue
                    else:
                        left_arg_str = foq_formula[begin + 1: i]
                else:
                    raise SyntaxError(
                        f"Query {foq_formula} is illegal for () delimiters")
                # address the only bracket case
                if begin == 0 and i == len(foq_formula) - 1:
                    return parse_top_foq_formula(left_arg_str,
                                                 z_obj=z_obj,
                                                 f_obj=f_obj,
                                                 binary_ops=binary_ops)

        elif c in binary_ops:  # handle the conjunction and disjunction
            if len(level_stack) == 0:  # only when at the top of the syntax tree
                top_binary_ops.append([i, c])

    if top_binary_ops:
        i, c = top_binary_ops[-1]
        return (binary_ops[c](), (foq_formula[:i], foq_formula[i + 1:]))

    # zero order decision: identify the zero order objects
    # consider two situations 'e' or '{x1, x2, ..., xn}'
    # you only need to initialize the variable class and assign the variable if necessary
    if foq_formula == 'e':
        return z_obj(), []
    if foq_formula[0] == '{' and foq_formula[-1] == '}':
        query = z_obj()
        try:
            query.entities = list(eval('[' + foq_formula[1:-1] + ']'))
        except:
            raise ValueError(
                f"fail to initialize f{foq_formula} as the value of zero order object")
        return query, []

    # first order decision: identify the first order objects
    # 'psub_foq_formula' or '[p1, p2, ..., pn]sub_foq_formula'
    # you should initialize the projection class, assign the possible projections if necessary
    # you should also return the argument for the
    if foq_formula[0] == 'p':
        query = f_obj()
        return query, [foq_formula[1:]]
    if foq_formula[0] == '[':  # trigger the second situation
        for i, c in enumerate(foq_formula):
            if c == ']':
                query = f_obj()
                try:
                    query.relations = list(eval(foq_formula[:i + 1]))
                except:
                    raise ValueError(
                        f"fail to initialize f{foq_formula} as the relation of first order object")
                return query, [foq_formula[i + 1:]]


def gen_foq_meta_formula(depth=0, max_depth=3, early_terminate=False):
    if depth >= max_depth or early_terminate:
        return "p(e)"

    et_choice = random.randint(0, 2)
    if et_choice == 0:
        et1, et2 = False, False
    elif et_choice == 1:
        et1, et2 = True, False
    elif et_choice == 2:
        et1, et2 = False, True

    t = random.randint(0, 3)
    if t == 0:
        return f"p({gen_foq_meta_formula(depth + 1, max_depth, early_terminate)})"
    elif t == 1:
        return f"({gen_foq_meta_formula(depth + 1, max_depth, et1)})&({gen_foq_meta_formula(depth + 1, max_depth, et2)})"
    elif t == 2:
        return f"({gen_foq_meta_formula(depth + 1, max_depth, et1)})|({gen_foq_meta_formula(depth + 1, max_depth, et2)})"
    elif t == 3:
        return f"({gen_foq_meta_formula(depth + 1, max_depth, et1)})-({gen_foq_meta_formula(depth + 1, max_depth, et2)})"