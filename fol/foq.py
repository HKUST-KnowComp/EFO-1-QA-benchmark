from abc import ABC, abstractclassmethod, abstractproperty
from typing import Tuple
from fol.appfoq import AppFOQEstimator
from fol.base import beta_query
from fol.sampler import read_indexing, load_data
import random
import collections
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
        self.objects = {}  # this is the intermediate objects during the sampling
        self.answer_set = {} # this is the answer set for answering deterministic

    @abstractproperty
    def ground_formula(self):
        pass

    @abstractproperty
    def meta_str(self):
        pass

    @abstractproperty
    def meta_formula(self):
        pass

    @abstractclassmethod
    def additive_ground(self, foq_formula, *args, **kwargs):
        pass

    @abstractclassmethod
    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        pass

    # @abstractclassmethod
    def deterministic_query(self, projection):
        #     Consider the first entity / relation
        #     """
        pass

    @abstractclassmethod
    def embedding_estimation(self, estimator: AppFOQEstimator):
        pass

    @abstractclassmethod
    def top_down_parse(self, *args, **kwargs):
        """ Parse meta or grounded formula
        """
        pass

    @abstractclassmethod
    def random_query(self, projection):
         pass

    def lift(self):
        """ Remove all intermediate objects, grounded entities (ZOO) and relations (FOO)
        """
        self.objects = {}


class VariableQ(FirstOrderQuery):
    def __init__(self):
        super().__init__()
        self.entities = []

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

    def embedding_estimation(self, estimator: AppFOQEstimator):
        return estimator.get_entity_embedding(self.entities)

    def lift(self):
        self.entities = []
        return super().lift()

    def top_down_parse(self, *args, **kwargs):
        return

    def deterministic_query(self, projection):
        return self.entities[0]

    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        if need_to_contain is False:
            essential_point = random.sample(set(reverse_projection.keys()) - {essential_point}, 1)[0]
        if need_to_contain is None and essential_point is None:
            essential_point = random.sample(set(reverse_projection.keys()), 1)[0]
        self.objects = {essential_point}
        return self.objects

    def random_query(self, projection):
        new_variable = random.sample(set(projection.keys()), 1)[0]
        self.objects.add(new_variable)
        return self.objects


class ProjectionQ(FirstOrderQuery):
    def __init__(self, q: FirstOrderQuery = None):
        super().__init__()
        self.operand_q = q
        self.relations = []
        self.rel = None

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

    def embedding_estimation(self, estimator: AppFOQEstimator):
        operand_emb = self.operand_q.embedding_estimation(estimator=estimator)
        return estimator.get_projection_embedding(self.relations, operand_emb)

    def lift(self):
        self.relations = []
        return super().lift()

    def top_down_parse(self, operand_str, **kwargs):
        assert len(operand_str) == 1
        operand_str = operand_str[0]
        obj, args = parse_top_foq_formula(foq_formula=operand_str, **kwargs)
        self.operand_q = obj
        self.operand_q.top_down_parse(args, **kwargs)

    def deterministic_query(self, projection):
        rel = self.relations[0]
        ans = self.operand_q.deterministic_query(projection)
        self.answer_set = set()
        for e in ans:
            self.answer_set.update(projection[ans][rel])
        return self.answer_set

    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        meeting_requirement = False  # whether have satisfied the essential_point condition
        # since the projection[next_point][self.rel] may contains essential_point even if not starting from it
        while meeting_requirement is False:
            if need_to_contain is False:
                now_point = random.sample(set(reverse_projection.keys()) - {essential_point}, 1)[0]
            else:
                if essential_point is None:
                    now_point = random.sample(set(reverse_projection.keys()), 1)[0]
                else:
                    now_point = essential_point
            self.rel = random.sample(reverse_projection[now_point].keys(), 1)[0]
            next_points = reverse_projection[now_point][self.rel]
            next_point = random.sample(next_points, 1)[0]   # find an incoming edge and a corresponding node
            meeting_requirement = True
            if need_to_contain is False and essential_point in projection[next_point][self.rel]:
                meeting_requirement = False
        f_object = self.operand_q.backward_sample(reverse_projection, projection, need_to_contain=True, essential_point=next_point)
        if None in f_object:
            raise ValueError
        self.objects = projection[next_point][self.rel]
        for entity in f_object.copy():
            if type(entity) == int:
                self.objects.update(projection[entity][self.rel])
        return self.objects

    def random_query(self, projection):
        variable = self.operand_q.random_query(projection)
        if len(variable) != 0:
            chosen_variable = random.sample(variable, 1)[0]
            self.rel = random.sample(projection[chosen_variable].keys(), 1)[0]
            self.objects = projection[chosen_variable][self.rel]
            for e in variable:
                self.objects.update(projection[e][self.rel])
            return self.objects
        else:
            return set(), set()


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

    def embedding_estimation(self, estimator: AppFOQEstimator):
        lemb = self.loperand_q.embedding_estimation(estimator=estimator)
        remb = self.roperand_q.embedding_estimation(estimator=estimator)
        return lemb, remb

    def lift(self):
        self.loperand_q.list()
        self.roperand_q.list()
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


class ConjunctionQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})&({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})&({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator):
        lemb, remb = super().embedding_estimation(estimator)
        return estimator.get_conjunction_embedding(lemb, remb)

    def deterministic_query(self, projection):
        l_ans = self.loperand_q.deterministic_query(projection)
        r_ans = self.roperand_q.deterministic_query(projection)
        return l_ans.intersection(r_ans)

    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        if essential_point is None:
            essential_point = random.sample(set(projection.keys()), 1)[0]
        if need_to_contain is False:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lobjs = self.loperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
                robjs = self.roperand_q.backward_sample(reverse_projection, projection)
            else:
                lobjs = self.loperand_q.backward_sample(reverse_projection, projection)
                robjs = self.roperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
        else:  # By default, choose a common start point.
            lobjs = self.loperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
            robjs = self.roperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
        self.objects = lobjs.intersection(robjs)
        return self.objects

    def random_query(self, projection):
        lobjs = self.loperand_q.random_query(projection)
        robjs = self.roperand_q.random_query(projection)
        self.objects = lobjs.intersection(robjs)
        return self.objects


class DisjunctionQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})|({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})|({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator):
        lemb, remb = super().embedding_estimation(estimator)
        return estimator.get_disjunction_embedding(lemb, remb)

    def deterministic_query(self, projection):
        l_ans = self.loperand_q.deterministic_query(projection)
        r_ans = self.roperand_q.deterministic_query(projection)
        return l_ans.union(r_ans)

    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        if essential_point is None:
            essential_point = random.sample(set(projection.keys()), 1)[0]
        if need_to_contain is False:
            lobjs = self.loperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
            robjs = self.roperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
        else:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lobjs = self.loperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
                robjs = self.roperand_q.backward_sample(reverse_projection, projection)
            else:
                lobjs = self.loperand_q.backward_sample(reverse_projection, projection)
                robjs = self.roperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
        self.objects = lobjs.union(robjs)
        return self.objects

    def random_query(self, projection):
        lobjs = self.loperand_q.random_query(projection)
        robjs = self.roperand_q.random_query(projection)
        self.objects = lobjs.union(robjs)
        return self.objects


class DifferenceQ(BinaryOps):
    def __init__(self, lq=None, rq=None):
        super().__init__(lq, rq)

    @property
    def ground_formula(self):
        return f"({self.loperand_q.ground_formula})-({self.roperand_q.ground_formula})"

    @property
    def meta_formula(self):
        return f"({self.loperand_q.meta_formula})-({self.roperand_q.meta_formula})"

    def embedding_estimation(self, estimator: AppFOQEstimator):
        lemb, remb = super().embedding_estimation(estimator)
        return estimator.get_difference_embedding(lemb, remb)

    def deterministic_query(self, projection):
        l_ans = self.loperand_q.deterministic_query(projection)
        r_ans = self.roperand_q.deterministic_query(projection)
        return l_ans - r_ans

    def backward_sample(self, reverse_projection, projection, need_to_contain: bool = None, essential_point=None):
        if need_to_contain is False:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lfobjs = self.loperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
                rfobjs = self.roperand_q.backward_sample(reverse_projection, projection)
            else:
                lfobjs = self.loperand_q.backward_sample(reverse_projection, projection,)
                rfobjs = self.roperand_q.backward_sample(
                    reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
        else:
            lfobjs = self.loperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=True, essential_point=essential_point)
            rfobjs = self.roperand_q.backward_sample(
                reverse_projection, projection, need_to_contain=False, essential_point=essential_point)
        self.objects = lfobjs - rfobjs
        return self.objects

    def random_query(self, projection):
        lfobjs = self.loperand_q.random_query(projection)
        rfobjs = self.roperand_q.random_query(projection)
        self.objects = lfobjs - rfobjs
        return self.objects


# you should specifiy the binary operator, zero order object, first order object
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
                          binary_ops=binary_ops) -> Tuple[
    FirstOrderQuery, Tuple[str]]:
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
            query.entities = list(eval(foq_formula))
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


if __name__ == '__main__':
    stanford_data_path = '../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
    projection_none = {}
    reverse_proection_none = {}
    for i in all_entity_dict.values():
        projection_none[i] = collections.defaultdict(set)
        reverse_proection_none[i] = collections.defaultdict(set)
    projection_train, reverse_projection_train = load_data('../datasets_knowledge_embedding/FB15k-237/train.txt',
                                                           all_entity_dict, all_relation_dict, projection_none,
                                                           reverse_proection_none)
    projection_valid, reverse_projection_valid = load_data('../datasets_knowledge_embedding/FB15k-237/valid.txt',
                                                           all_entity_dict, all_relation_dict, projection_train,
                                                           reverse_projection_train)

    for name in beta_query:
        query_structure = beta_query[name]
        ansclass = parse_foq_formula(foq_formula=query_structure)
        ans_objects = ansclass.backward_sample(reverse_projection_train, projection_train)
        ans_2 = ansclass.random_query(projection_train)
        print(ans_objects, ans_2)

