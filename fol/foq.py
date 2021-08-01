from pickle import APPEND
import random
from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

import torch
from fol.appfoq import AppFOQEstimator, IntList

"""
First Order Query (FOQ) is a conceptual idea without any implementation
First order formula is a string formulation of the FOQ

Grammar of the first order formula
Formula := FirstOrderObject(Formula)
            OR FirstOrderObject(ZeroOrderObject)
            OR Formula & Formula
            OR Formula - Formula
            OR Formula | Formula
            OR (Formula)
FirstOrderObject := p
                    OR [IntegerList]
ZeroOrderObject := e
                    OR {IntegerList}
IntegerList := Integer
                OR IntegerList,IntegerList
Remark:
    1. If a formula only includes ()&|-ep, it is a meta formula
    2. Otherwise, it is a grounded formula
    3. If a formula doesn't contain any p and e, it is a fully grounded formula.
        we can do `exact_query` and `embedding_estimation`
        3.1 `exact_query` and `embedding_estimation` has different complexity.
    4. If a formula is not fully grounded, we can do `backward_sample`
    5. Meta formula defines an equivalent class of our interest.
        We allow data parallelation of formulas inside the same equivalent class.
        We can parallelize `embedding_estimation` inside the same equivalent class.
        To prepare the data for parallel computing, we can do `additive_ground`.
"""


class FirstOrderQuery(ABC):
    def __init__(self):
        # self.objects = {}  # this is the intermediate objects during the sampling
        # FIXME: remove self.object since it is non-necessary
        # self.answer_set = {} # this is the answer set for answering deterministic
        pass

    @property
    @abstractmethod
    def ground_formula(self) -> str:
        pass

    @property
    @abstractmethod
    def meta_str(self) -> str:
        pass

    @property
    @abstractmethod
    def meta_formula(self) -> str:
        pass

    @abstractmethod
    def additive_ground(self, foq_formula, *args, **kwargs):
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
    def top_down_parse(self, *args, **kwargs):
        """ Parse meta or grounded formula
        """
        pass

    @abstractmethod
    def random_query(self, projs, cumulative: bool = False):
        pass

    @abstractmethod
    def lift(self):
        """ Remove all intermediate objects, grounded entities (ZOO) and relations (FOO)
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


class VariableQ(FirstOrderQuery):
    """
    The `self.objects` inherented from the parent class is not used
    """

    def __init__(self):
        super().__init__()
        self.entities = []
        self.tentities = None
        self.device = 'cpu'

    @property
    def ground_formula(self):
        if self.entities:
            return "{" + ",".join(str(e) for e in self.entities) + "}"
        else:
            return "e"

    @property
    def meta_formula(self):
        return "e"

    @property
    def meta_str(self):
        return type(self).__name__

    def additive_ground(self, foq_formula, *args, **kwargs):
        obj, args = parse_top_foq_formula(foq_formula=foq_formula, **kwargs)
        if isinstance(obj, type(self)):
            assert len(args) == 0
            self.entities += obj.entities
        else:
            raise ValueError(
                f"formula {foq_formula} is not in the same equivalence meta query class {self.meta_formula}")

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        if self.tentities is None:
            self.tentities = torch.tensor(self.entities).to(self.device)
        if batch_indices:
            ent = self.tentities[torch.tensor(batch_indices)]
        else:
            ent = self.tentities
        return estimator.get_entity_embedding(ent.to(self.device))

    def lift(self):
        self.entities = []
        self.tentities = None
        return super().lift()

    def top_down_parse(self, *args, **kwargs):
        return

    def deterministic_query(self, projection):  # TODO: change to return a list of set
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


class ProjectionQ(FirstOrderQuery):
    """
    `self.relations` describes the relation ids by the KG
    """

    def __init__(self, q: FirstOrderQuery = None):
        super().__init__()
        self.operand_q = q
        self.relations = []
        self.trelations = None
        self.device = 'cpu'

    @property
    def ground_formula(self):
        if self.relations:
            return "[" + ",".join(str(r) for r in self.relations) + "]" + f"({self.operand_q.ground_formula})"
        else:
            return f"p({self.operand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"p({self.operand_q.meta_formula})"

    @property
    def meta_str(self):
        return f"{type(self).__name__}({self.operand_q.meta_str})"

    def additive_ground(self, foq_formula, *args, **kwargs):
        obj, args = parse_top_foq_formula(foq_formula=foq_formula, **kwargs)
        if isinstance(obj, type(self)):
            self.relations += obj.relations
            assert len(args) == 1
            self.operand_q.additive_ground(args[0])
        else:
            raise ValueError(
                f"formula {foq_formula} is not in the same equivalence meta query class {self.meta_formula}")

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        if self.trelations is None:
            self.trelations = torch.tensor(self.relations).to(self.device)
        if batch_indices:
            rel = self.trelations[torch.tensor(batch_indices)]
        else:
            rel = self.trelations
        operand_emb = self.operand_q.embedding_estimation(estimator, batch_indices)
        return estimator.get_projection_embedding(rel.to(self.device), operand_emb)

    def lift(self):
        self.relations = []
        self.trelations = None
        return super().lift()

    def top_down_parse(self, operand_str, **kwargs):
        assert len(operand_str) == 1
        operand_str = operand_str[0]
        obj, args = parse_top_foq_formula(foq_formula=operand_str, **kwargs)
        self.operand_q = obj
        self.operand_q.top_down_parse(args, **kwargs)

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
                    cursor = random.sample(set(rprojs.keys()) - {keypoint}, 1)[0]
            else:
                cursor = random.sample(set(rprojs.keys()), 1)[0]

            relation = random.sample(rprojs[cursor].keys(), 1)[0]
            parents = rprojs[cursor][relation]
            parent = random.sample(parents, 1)[0]  # find an incoming edge and a corresponding node

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


class BinaryOps(FirstOrderQuery):
    def __init__(self, lq: FirstOrderQuery, rq: FirstOrderQuery):
        self.loperand_q, self.roperand_q = lq, rq

    @property
    def meta_str(self):
        return f"{type(self).__name__}({self.loperand_q.meta_str}, {self.roperand_q.meta_str})"

    def additive_ground(self, foq_formula, *args, **kwargs):
        obj, args = parse_top_foq_formula(foq_formula, **kwargs)
        if isinstance(obj, type(self)):
            assert len(args) == 2
            largs, rargs = args
            self.loperand_q.additive_ground(largs, **kwargs)
            self.roperand_q.additive_ground(rargs, **kwargs)
        else:
            raise ValueError(
                f"formula {foq_formula} is not in the same equivalence meta query class {self.meta_formula}")

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        lemb = self.loperand_q.embedding_estimation(estimator, batch_indices)
        remb = self.roperand_q.embedding_estimation(estimator, batch_indices)
        return lemb, remb

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


class ConjunctionQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})&({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})&({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        lemb, remb = super().embedding_estimation(estimator, batch_indices)
        return estimator.get_conjunction_embedding(lemb, remb)

    def deterministic_query(self, projs):
        l_result = self.loperand_q.deterministic_query(projs)
        r_result = self.roperand_q.deterministic_query(projs)
        return l_result.intersection(r_result)

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
                    robjs = self.roperand_q.backward_sample(projs, rprojs, cumulative=cumulative)
                else:
                    lobjs = self.loperand_q.backward_sample(projs, rprojs, cumulative=cumulative)
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


class DisjunctionQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})|({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})|({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        lemb, remb = super().embedding_estimation(estimator, batch_indices)
        return estimator.get_disjunction_embedding(lemb, remb)

    def deterministic_query(self, projs):
        l_result = self.loperand_q.deterministic_query(projs)
        r_result = self.roperand_q.deterministic_query(projs)
        return l_result.union(r_result)

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
                    robjs = self.roperand_q.backward_sample(projs, rprojs, cumulative=cumulative)
                else:
                    lobjs = self.loperand_q.backward_sample(projs, rprojs, cumulative=cumulative)
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


class DifferenceQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})-({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})-({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices=None):
        lemb, remb = super().embedding_estimation(estimator, batch_indices)
        return estimator.get_difference_embedding(lemb, remb)

    def deterministic_query(self, projs):
        l_result = self.loperand_q.deterministic_query(projs)
        r_result = self.roperand_q.deterministic_query(projs)
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


binary_ops = {
    '|': DisjunctionQ,
    '&': ConjunctionQ,
    '-': DifferenceQ
}

grammar_class = {
    'z_obj': VariableQ,
    'f_obj': ProjectionQ,
    'binary_ops': binary_ops
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


def parse_foq_formula(foq_formula: str, grammar_class=grammar_class) -> FirstOrderQuery:
    """ This function parse a first order query string (with or without instantiation) into nested classes
    """
    obj, args = parse_top_foq_formula(foq_formula, **grammar_class)
    obj.top_down_parse(args, **grammar_class)
    return obj


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
