import collections
import copy
import json
import random
from itertools import product
from abc import ABC, abstractmethod
from typing import List, Tuple, TypedDict
from typing import Union as TUnion
from numpy.core.arrayprint import SubArrayFormat
from collections import Counter, defaultdict
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
        - 'n' for negation,     1 argument: list of negations
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
        self.latent_embedding = None
        pass

    @abstractmethod
    def sort_sub(self):
        """
        Sort the sub query by their formula in alphabetical order.
        """
        pass

    @property
    @abstractmethod
    def formula(self) -> str:
        """
        Where we store the structure information
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
    def backward_sample(self, projs, rprojs, requirement=None,
                        cumulative: bool = False, meaningful_difference: bool = False, **kwargs):
        """
        A function used to ground a query, the backward sampling strategy is used to ensure that there are always an
        answer of this query, specifically, queries like 3i tend to have no answers if you use random_query rather than
        backward_sample.
        Requirement is a defaultdict(set)
        If the meaningful_difference is False, the requirement only have 1 key: 'must include' or 'must exclude'.
        If the meaningful_difference is True, the requirement may have 4 keys: 'must include', 'must exclude',
        'optional include', 'optional exclude', the priority follows this order.
        All possibilities are 'I' / 'E+O_I' / 'E' / 'O_I+O_E' / 'O_I' / 'O_E', those are short for requirement keys.
        All requirement have at most one element given the new assignment of requirement in intersection.
        """
        pass

    @abstractmethod
    def deterministic_query(self, projection):
        #     Consider the first entity / relation
        pass

    @abstractmethod
    def embedding_estimation(self, estimator: AppFOQEstimator, batch_indices,
                             **kwargs):
        pass

    # @abstractmethod
    def _embedding_optimization(self, estimator: AppFOQEstimator,
                                batch_indices, **kwargs):
        """
        This function is called for internal usage
        it first initializes the latent embeddings of eqch fosq instance
        then computes the loss function
        For entity query, the loss is None,
        For entity query, the loss is not None
        For projection, we have the t norm
        """
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


def embedding_estimation_with_optimization(fosq: FirstOrderSetQuery,
                                           estimator,
                                           optimizer_name="SGD",
                                           num_steps=10,
                                           optimizer_args={"lr": 1}):
    for i in range(10):
        loss, parameters = fosq._embedding_optimization(estimator)
        optimizer = getattr(torch.optim, optimizer_name)(
            parameters, **optimizer_args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fosq.latent_embedding


class Entity(FirstOrderSetQuery):
    """ A singleton set of entities
    """
    __o__ = 'e'

    def __init__(self):
        super().__init__()
        self.entities = []
        self.tentities = None
        self.device = 'cpu'
        
    def sort_sub(self):
        pass
        
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
                             batch_indices=None,
                             **kwargs):
        if self.tentities is None:
            self.tentities = torch.tensor(self.entities).to(self.device)
        if batch_indices:
            ent = self.tentities[torch.tensor(batch_indices)]
        else:
            ent = self.tentities
        return estimator.get_entity_embedding(ent, **kwargs)

    def _embedding_optimization(self,
                                estimator: AppFOQEstimator,
                                batch_indices=None,
                                **kwargs):
        self.latent_embedding = self.embedding_estimation(
            estimator=estimator,
            batch_indices=batch_indices)
        return None, [],

    def lift(self):
        self.entities = []
        self.tentities = None

    def deterministic_query(self, args, **kwargs):
        # TODO: change to return a list of set
        return {self.entities[0]}

    def backward_sample(self, projs, rprojs, requirement=None, cumulative=False, **kwargs):
        if requirement:
            if requirement['must include']:
                new_entity = list(requirement['must include'])
            else:
                new_entity = list(random.sample(set(projs.keys()) - {requirement['mustnot include']}, 1))
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
        else:
            self.tentities = self.tentities.to(device)
        print(f'move variable object in {id(self)} to device {device}')


class Negation(FirstOrderSetQuery):
    __o__ = 'n'

    def __init__(self, q: FirstOrderSetQuery = None):
        super().__init__()
        self.query = q
        
    def sort_sub(self):
        self.query.sort_sub()

    @property
    def formula(self):
        return f"(n,{self.query.formula})"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': json.loads(self.query.dumps)
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, sub_dobject = dobject['o'], dobject['a']
        assert obj == self.__o__
        self.query.additive_ground(sub_dobject)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None, **kwargs):
        operand_emb = self.query.embedding_estimation(estimator,
                                                      batch_indices,
                                                      **kwargs)
        return estimator.get_negation_embedding(operand_emb, **kwargs)

    def lift(self):
        self.query.lift()

    def deterministic_query(self, projection):
        ans = projection.keys() - self.query.deterministic_query(projection)
        return ans

    def backward_sample(self, projs, rprojs, requirement: bool = None, cumulative=False,
                        meaningful_difference: bool = False, **kwargs):
        # assert meaningful_difference == False  This is not true only in DM-like queries
        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}
        new_requirement = defaultdict(set)
        new_requirement['must include'] = requirement['must exclude']
        new_requirement['must exclude'] = requirement['must include']
        return projs.keys() - self.query.backward_sample(projs, rprojs, new_requirement, cumulative,
                                                         meaningful_difference, **kwargs)

    def random_query(self, projs, cumulative=False):
        ans = projs.keys() - self.query.random_query(projs, cumulative)
        return ans

    def check_ground(self) -> int:
        return self.query.check_ground()

    def to(self, device):
        self.query.to(device)


class Projection(FirstOrderSetQuery):
    """
    `self.relations` describes the relation ids by the KG
    """
    __o__ = 'p'

    def __init__(self, q: FirstOrderSetQuery = None):
        super().__init__()
        self.query = q
        self.relations = []
        self.trelations = None
        self.device = 'cpu'
        
    def sort_sub(self):
        self.query.sort_sub()

    @property
    def formula(self):
        return f"(p,{self.query.formula})"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': [self.relations, json.loads(self.query.dumps)]
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, (relation_list, sub_dobject) = dobject['o'], dobject['a']
        assert obj == self.__o__
        assert all(isinstance(i, int) for i in relation_list)
        self.relations.extend(relation_list)
        self.query.additive_ground(sub_dobject)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None,
                             **kwargs):
        if self.trelations is None:
            self.trelations = torch.tensor(self.relations).to(self.device)
        if batch_indices:
            rel = self.trelations[torch.tensor(batch_indices)]
        else:
            rel = self.trelations
        operand_emb = self.query.embedding_estimation(estimator,
                                                      batch_indices,
                                                      **kwargs)
        return estimator.get_projection_embedding(rel,
                                                  operand_emb,
                                                  **kwargs)

    def lift(self):
        self.relations = []
        self.trelations = None
        self.query.lift()

    def deterministic_query(self, projs):
        rel = self.relations[0]
        result = self.query.deterministic_query(projs)
        answer = set()
        for e in result:
            answer.update(projs[e][rel])
        return answer

    def backward_sample(self, projs, rprojs, requirement=None, cumulative=False, meaningful_difference: bool = False,
                        **kwargs):
        # since the projection[next_point][self.rel] may contains essential_point even if not starting from it
        # This issue can not be totally solved since the p_object contains other entity than parent
        # TODO: maybe solve this by not considering in edge type, but seems unfavourable in meaningful difference.
        def find_exlusion(projs, rprojs, exclude_point):
            while True:
                cursor = random.sample(rprojs.keys() - exclude_point, 1)[0]
                relation = random.sample(rprojs[cursor].keys(), 1)[0]
                parents = rprojs[cursor][relation]
                parent = random.sample(parents, 1)[0]
                if not exclude_point.issubset(projs[parent][relation]):
                    break
            return parent, relation

        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}

        if requirement['must include']:
            cursor = list(requirement['must include'])[0]
            relation = random.sample(rprojs[cursor].keys(), 1)[0]
            parents = rprojs[cursor][relation]
            # find an incoming edge and a corresponding node
            parent = random.sample(parents, 1)[0]
        elif requirement['must exclude']:
            if requirement['optional include']:
                cursor = list(requirement['optional include'])[0]
                for relation in rprojs[cursor].keys():
                    parents = rprojs[cursor][relation]
                    for parent in parents:
                        if not requirement['must exclude'].issubset(projs[parent][relation]):
                            break
                    else:
                        continue
                    break
                else:
                    parent, relation = find_exlusion(projs, rprojs, requirement['must exclude'])
            else:
                parent, relation = find_exlusion(projs, rprojs, requirement['must exclude'])
        elif requirement['optional include']:
            cursor = list(requirement['optional include'])[0]
            for relation in rprojs[cursor].keys():
                parents = rprojs[cursor][relation]
                for parent in parents:
                    if not requirement['optional exclude'].issubset(projs[parent][relation]):
                        break
                else:
                    continue
                break
        elif requirement['optional exclude']:
            parent, relation = find_exlusion(projs, rprojs, requirement['optional exclude'])
        else:  # requirement is empty defaultdict(set)
            parent, relation = find_exlusion(projs, rprojs, {-1})

        new_requirement = defaultdict(set)
        new_requirement['must include'] = {parent}
        p_object = self.query.backward_sample(projs, rprojs, requirement=new_requirement, cumulative=cumulative,
                                              meaningful_difference=meaningful_difference, **kwargs)
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
        variable = self.query.random_query(projs, cumulative)
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
        n_inst = self.query.check_ground()
        assert len(self.relations) == n_inst
        return n_inst

    def to(self, device):
        self.device = device
        if self.trelations is None:
            self.trelations = torch.tensor(self.relations).to(device)
        else:
            self.trelations = self.trelations.to(device)
        print(f'move projection object in {id(self)} to device {device}')
        self.query.to(device)


class MultipleSetQuery(FirstOrderSetQuery):
    def __init__(self, *queries: List[FirstOrderSetQuery]):
        self.sub_queries = queries
        
    def sort_sub(self):
        self.sub_queries = sorted(self.sub_queries, key=lambda q: q.formula)

    @property
    def dumps(self):
        self.sort_sub()
        dobject = {
            'o': self.__o__,
            'a': [
                json.loads(subq.dumps) for subq in self.sub_queries
            ]
        }
        return json.dumps(dobject)

    @property
    def formula(self):
        self.sort_sub()
        if len(self.sub_queries) > 2:
            symb = self.__o__.upper()
        else:
            symb = self.__o__
        return "({},{})".format(
            symb,
            ",".join(q.formula for q in self.sub_queries)
        )

    def additive_ground(self, dobject: Dobject):
        obj, sub_dobjects = dobject['o'], dobject['a']
        assert obj == self.__o__
        assert len(self.sub_queries) == len(sub_dobjects)
        for q, dobj in zip(self.sub_queries, sub_dobjects):
            q.additive_ground(dobj)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None,
                             **kwargs):
        return [q.embedding_estimation(estimator, batch_indices, **kwargs)
                for q in self.sub_queries]

    def lift(self):
        for query in self.sub_queries:
            query.lift()

    def check_ground(self):
        checked = set(q.check_ground() for q in self.sub_queries)
        assert len(checked) == 1
        return list(checked)[0]

    def to(self, device):
        for q in self.sub_queries:
            q.to(device)


class Intersection(MultipleSetQuery):
    __o__ = 'i'

    def __init__(self, *queries: List[FirstOrderSetQuery]):
        super().__init__(*queries)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None,
                             **kwargs):
        embed_list = super().embedding_estimation(estimator,
                                                  batch_indices,
                                                  **kwargs)
        return estimator.get_conjunction_embedding(embed_list, **kwargs)

    def deterministic_query(self, projs):
        return set.intersection(
            *(q.deterministic_query(projs) for q in self.sub_queries)
        )

    def backward_sample(self, projs, rprojs, requirement=None, cumulative: bool = False,
                        meaningful_difference: bool = False, **kwargs):
        sub_obj_list, pos_obj_list, neg_obj_list = [], [], []
        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}
        positive_subqueries, neg_subqueries = [], []
        for sub_query in self.sub_queries:
            if sub_query.__o__ == 'n':
                neg_subqueries.append(sub_query.query)
            else:
                positive_subqueries.append(sub_query)
        if meaningful_difference and len(positive_subqueries) > 0:
            positive_requirement = defaultdict(set)
            positive_requirement['must include'] = requirement['must include']
            positive_requirement['optional include'] = requirement['optional include']
            positive_choose_requirement = copy.deepcopy(requirement)
            choose_formula = random.randint(0, len(positive_subqueries) - 1)
            for i in range(len(positive_subqueries)):
                if i == choose_formula:
                    pos_objs = positive_subqueries[i].backward_sample(projs, rprojs, positive_choose_requirement,
                                                                      cumulative, meaningful_difference, **kwargs)
                else:
                    pos_objs = positive_subqueries[i].backward_sample(projs, rprojs, positive_requirement,
                                                                      cumulative, meaningful_difference, **kwargs)
                pos_obj_list.append(pos_objs)
            all_pos_objs = set.intersection(*pos_obj_list)
            negative_requirement = defaultdict(set)
            negative_requirement['must exclude'] = requirement['must include']
            negative_requirement['optional exclude'] = requirement['optional include']
            optional_exclude_set = all_pos_objs - requirement['must include'] - requirement['optional include']
            max_exclude_num = min(len(optional_exclude_set), len(neg_subqueries))
            negative_choose_formulas = random.sample(list(range(len(neg_subqueries))), max_exclude_num)
            optional_exclude_list = list(optional_exclude_set)
            exclude_element_list = random.sample(optional_exclude_list, max_exclude_num)
            exclude_ordinal = 0
            for i in range(len(neg_subqueries)):
                if i in negative_choose_formulas:
                    specific_negative_requirement = copy.deepcopy(negative_requirement)
                    specific_negative_requirement['optional include'] = {exclude_element_list[exclude_ordinal]}
                    exclude_ordinal += 1
                    neg_objs = neg_subqueries[i].backward_sample(
                        projs, rprojs, specific_negative_requirement, cumulative, meaningful_difference, **kwargs)
                else:
                    neg_objs = neg_subqueries[i].backward_sample(projs, rprojs, negative_requirement, cumulative,
                                                                 meaningful_difference, **kwargs)
                all_pos_objs = all_pos_objs - neg_objs
            return all_pos_objs
        else:
            new_requirement = copy.deepcopy(requirement)
            if requirement['must include']:
                for sub_query in self.sub_queries:
                    sub_objs = sub_query.backward_sample(projs, rprojs, new_requirement, cumulative,
                                                         meaningful_difference, **kwargs)
                    sub_obj_list.append(sub_objs)
            else:
                choose_formula = random.randint(0, len(self.sub_queries) - 1)
                for i in range(len(self.sub_queries)):
                    if i != choose_formula:
                        sub_objs = self.sub_queries[i].backward_sample(
                            projs, rprojs, requirement=None, cumulative=cumulative,
                            meaningful_difference=meaningful_difference, **kwargs)
                    else:
                        sub_objs = self.sub_queries[i].backward_sample(projs, rprojs, new_requirement, cumulative,
                                                                       meaningful_difference, **kwargs)
                    sub_obj_list.append(sub_objs)
            return set.intersection(*sub_obj_list)

    def random_query(self, projs, cumulative=False):
        sub_obj_list = []
        for query in self.sub_queries:
            sub_obj = query.random_query(projs, cumulative=cumulative)
            sub_obj_list.append(sub_obj)
        return set.intersection(*sub_obj_list)


class Union(MultipleSetQuery):
    __o__ = 'u'

    def __init__(self, *queries: List[FirstOrderSetQuery]):
        super().__init__(*queries)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None,
                             **kwargs):
        embed_list = super().embedding_estimation(estimator,
                                                  batch_indices,
                                                  **kwargs)
        return estimator.get_disjunction_embedding(embed_list, **kwargs)

    def deterministic_query(self, projs):
        return set.union(
            *(q.deterministic_query(projs) for q in self.sub_queries)
        )

    def backward_sample(self, projs, rprojs, requirement=None, cumulative=False,
                        meaningful_difference: bool = False, **kwargs):
        sub_obj_list = []
        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}
        normal_requirement = defaultdict(set)
        normal_requirement['must exclude'] = requirement['must exclude']
        normal_requirement['optional exclude'] = requirement['optional exclude']
        if requirement['must include'] or requirement['optional include']:
            choose_formula_num = random.randint(0, len(self.sub_queries) - 1)
            specific_requirement = copy.deepcopy(requirement)
            for i in range(len(self.sub_queries)):
                if i == choose_formula_num:
                    sub_objs = self.sub_queries[i].backward_sample(projs, rprojs, specific_requirement, cumulative,
                                                                   meaningful_difference, **kwargs)
                    sub_obj_list.append(sub_objs)
                else:
                    sub_objs = self.sub_queries[i].backward_sample(projs, rprojs, normal_requirement, cumulative,
                                                                   meaningful_difference, **kwargs)
                    sub_obj_list.append(sub_objs)
        else:
            for query in self.sub_queries:
                sub_objs = query.backward_sample(projs, rprojs, normal_requirement, cumulative,
                                                 meaningful_difference, **kwargs)
                sub_obj_list.append(sub_objs)
        return set.union(*sub_obj_list)

    def random_query(self, projs, cumulative=False):
        sub_obj_list = []
        for query in self.sub_queries:
            sub_objs = query.random_query(projs, cumulative=cumulative)
            sub_obj_list.append(sub_objs)
        return set.union(*sub_obj_list)


class Difference(FirstOrderSetQuery):
    __o__ = 'd'

    def __init__(self, lq: FirstOrderSetQuery, rq: FirstOrderSetQuery):
        self.lquery = lq
        self.rquery = rq

    def sort_sub(self):
        self.lquery.sort_sub()
        self.rquery.sort_sub()

    @property
    def formula(self):
        return f"(d,{self.lquery.formula},{self.rquery.formula})"

    @property
    def dumps(self):
        dobject = {
            'o': self.__o__,
            'a': [json.loads(self.lquery.dumps), json.loads(self.rquery.dumps)]
        }
        return json.dumps(dobject)

    def additive_ground(self, dobject: Dobject):
        obj, (ldobj, rdobj) = dobject['o'], dobject['a']
        assert obj == self.__o__
        self.lquery.additive_ground(ldobj)
        self.rquery.additive_ground(rdobj)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None, **kwargs):
        lemb = self.lquery.embedding_estimation(estimator,
                                                batch_indices,
                                                **kwargs)
        remb = self.rquery.embedding_estimation(estimator,
                                                batch_indices,
                                                **kwargs)
        return estimator.get_difference_embedding(lemb, remb, **kwargs)

    def deterministic_query(self, projs):
        l_result = self.lquery.deterministic_query(projs)
        r_result = self.rquery.deterministic_query(projs)
        return l_result - r_result

    def backward_sample(self, projs, rprojs, requirement=None, cumulative=False,
                        meaningful_difference: bool = False, **kwargs):
        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}
        sub_obj_list, neg_obj_list = [], []
        lquery, rquery = self.sub_queries[0], self.sub_queries[1]
        negative_requirement = defaultdict(set)
        negative_requirement['must exclude'] = requirement['must include']
        if meaningful_difference:
            positive_choose_requirement = copy.deepcopy(requirement)
            pos_objs = lquery.backward_sample(projs, rprojs, positive_choose_requirement,
                                              cumulative, meaningful_difference, **kwargs)
            negative_requirement['optional exclude'] = requirement['optional include']
            optional_exclude_set = pos_objs - requirement['must include'] - requirement['optional include']
            exclude_element = random.sample(optional_exclude_set, 1)
            specific_negative_requirement = copy.deepcopy(negative_requirement)
            specific_negative_requirement['optional include'] = set(exclude_element)
            neg_objs = rquery.backward_sample(
                projs, rprojs, specific_negative_requirement, cumulative, meaningful_difference, **kwargs)
            pos_objs = pos_objs - neg_objs
            return pos_objs
        else:
            positive_requirement = copy.deepcopy(requirement)
            negative_requirement['must include'] = requirement['must exclude']
            if requirement['must include']:
                lobjs = lquery.backward_sample(projs, rprojs, positive_requirement,
                                               cumulative, meaningful_difference, **kwargs)
                robjs = rquery.backward_sample(projs, rprojs, negative_requirement,
                                               cumulative, meaningful_difference, **kwargs)
            else:
                choose_lr = random.randint(0, 1)
                if choose_lr:
                    lobjs = lquery.backward_sample(projs, rprojs, positive_requirement,
                                                   cumulative, meaningful_difference, **kwargs)
                    robjs = rquery.backward_sample(projs, rprojs, requirement=None, cumulative=cumulative,
                                                   meaningful_difference=meaningful_difference, **kwargs)
                else:
                    lobjs = lquery.backward_sample(projs, rprojs, requirement=None, cumulative=cumulative,
                                                   meaningful_difference=meaningful_difference, **kwargs)
                    robjs = rquery.backward_sample(projs, rprojs, negative_requirement, cumulative,
                                                   meaningful_difference, **kwargs)
            lobjs = lobjs - robjs
            return lobjs

    def random_query(self, projs, cumulative=False):
        lobjs = self.lquery.random_query(projs, cumulative)
        robjs = self.rquery.random_query(projs, cumulative)
        return lobjs - robjs

    def lift(self):
        self.lquery.lift()
        self.rquery.lift()

    def check_ground(self) -> int:
        n1 = self.lquery.check_ground()
        n2 = self.rquery.check_ground()
        assert n1 == n2
        return n1

    def to(self, device):
        self.lquery.to(device)
        self.rquery.to(device)


class Multiple_Difference(MultipleSetQuery):
    __o__ = 'D'

    def __init__(self, *queries: List[FirstOrderSetQuery]):
        super().__init__(*queries)
        
    def sort_sub(self):
        lquery, rqueries = self.sub_queries[0], self.sub_queries[1:]
        lquery.sort_sub()
        rqueries = sorted(rqueries, key=lambda q: q.formula)
        self.sub_queries = [lquery] + rqueries

    @property
    def formula(self):
        self.sort_sub()
        return "({},{})".format(
            self.__o__,
            ",".join(subq.formula for subq in self.sub_queries)
        )

    @property
    def dumps(self):
        self.sort_sub()
        dobject = {
            'o': self.__o__,
            'a': [
                json.loads(subq.dumps) for subq in self.sub_queries
            ]
        }
        return json.dumps(dobject)

    def embedding_estimation(self,
                             estimator: AppFOQEstimator,
                             batch_indices=None, **kwargs):
        embed_list = super().embedding_estimation(estimator,
                                                  batch_indices,
                                                  **kwargs)
        return estimator.get_multiple_difference_embedding(embed_list, batch_indices, **kwargs)

    def deterministic_query(self, projs):
        lquery, rqueries = self.sub_queries[0], self.sub_queries[1:]
        ans_excluded = set.union(*[sub_query.deterministic_query(projs) for sub_query in rqueries])
        ans_origin = lquery.deterministic_query(projs)
        return ans_origin - ans_excluded

    def backward_sample(self, projs, rprojs, requirement=None, cumulative=False,
                        meaningful_difference: bool = False, **kwargs):
        sub_obj_list, neg_obj_list = [], []
        lquery, rqueries = self.sub_queries[0], self.sub_queries[1:]
        if not requirement:
            requirement = defaultdict(set)
            requirement['must include'] = {random.randrange(0, len(projs.keys()))}
        positive_choose_requirement = copy.deepcopy(requirement)
        negative_requirement = defaultdict(set)
        negative_requirement['must exclude'] = requirement['must include']
        if meaningful_difference:
            negative_requirement['optional exclude'] = requirement['optional include']
            pos_objs = lquery.backward_sample(projs, rprojs, positive_choose_requirement,
                                              cumulative, meaningful_difference, **kwargs)

            optional_exclude_set = pos_objs - requirement['must include'] - requirement['optional include']
            max_exclude_num = min(len(optional_exclude_set), len(rqueries))
            negative_choose_formulas = random.sample(list(range(len(rqueries))), max_exclude_num)
            optional_exclude_list = list(optional_exclude_set)
            exclude_element_list = random.sample(optional_exclude_list, max_exclude_num)
            exclude_ordinal = 0
            for i in range(len(rqueries)):
                if i in negative_choose_formulas:
                    specific_negative_requirement = copy.deepcopy(negative_requirement)
                    specific_negative_requirement['optional include'] = {exclude_element_list[exclude_ordinal]}
                    exclude_ordinal += 1
                    neg_objs = rqueries[i].backward_sample(
                        projs, rprojs, specific_negative_requirement, cumulative, meaningful_difference, **kwargs)
                else:
                    neg_objs = rqueries[i].backward_sample(projs, rprojs, negative_requirement, cumulative,
                                                           meaningful_difference, **kwargs)
                pos_objs = pos_objs - neg_objs
            return pos_objs
        else:
            negative_requirement['must include'] = requirement['must exclude']
            if requirement['must include']:
                lobjs = lquery.backward_sample(projs, rprojs, positive_choose_requirement,
                                               cumulative, meaningful_difference, **kwargs)
                for rquery in rqueries:
                    robjs = rquery.backward_sample(projs, rprojs, negative_requirement,
                                                   cumulative, meaningful_difference, **kwargs)
                    lobjs = lobjs - robjs
            else:
                choose_lr = random.randint(0, 1)
                if choose_lr:
                    lobjs = lquery.backward_sample(projs, rprojs, positive_choose_requirement,
                                                   cumulative, meaningful_difference, **kwargs)
                    for rquery in rqueries:
                        robjs = rquery.backward_sample(projs, rprojs, requirement=None, cumulative=cumulative,
                                                       meaningful_difference=meaningful_difference, **kwargs)
                        lobjs = lobjs - robjs
                else:
                    choose_formula = random.randrange(0, len(rqueries))
                    lobjs = lquery.backward_sample(projs, rprojs, requirement=None, cumulative=cumulative,
                                                   meaningful_difference=meaningful_difference, **kwargs)
                    for i in range(len(rqueries)):
                        if i == choose_formula:
                            robjs = rqueries[i].backward_sample(projs, rprojs, negative_requirement, cumulative,
                                                                meaningful_difference, **kwargs)
                        else:
                            robjs = rqueries[i].backward_sample(projs, rprojs, requirement=None, cumulative=cumulative,
                                                                meaningful_difference=meaningful_difference, **kwargs)
                        lobjs = lobjs - robjs
            return lobjs

    def random_query(self, projs, cumulative=False):
        robj_list = []
        lquery, rqueries = self.sub_queries[0], self.sub_queries[1:]
        lobjs = lquery.random_query(projs, cumulative=cumulative)
        for query in rqueries:
            robjs = query.random_query(projs, cumulative=cumulative)
            robj_list.append(robjs)
        return lobjs - set.union(*robj_list)


ops_dict = {
    'e': Entity,
    'n': Negation,
    'p': Projection,
    'i': Intersection,
    'I': Intersection,
    'u': Union,
    'U': Union,
    'd': Difference,
    'D': Multiple_Difference
}


def parse_formula(fosq_formula: str) -> FirstOrderSetQuery:
    """ A new function to parse first-order set query string
    """
    cached_objects = {}
    cached_subranges = {}
    todo_ranges = []

    def identify_range(i, j):
        """ i, and j is the index of ( and ) respectively
        identify the information contained in the range
        return
            ops: operational string
            sub_range_list: a list of sub ranges
        """
        ops = fosq_formula[i + 1]
        level_stack = []
        sub_range_list = []
        for k in range(i + 1, j):
            if fosq_formula[k] == '(':
                level_stack.append(k)
            elif fosq_formula[k] == ')':
                begin = level_stack.pop(-1)
                if len(level_stack) == 0:
                    sub_range_list.append((begin, k))
        if ops == 'e':
            assert len(sub_range_list) == 0
        elif ops in 'pn':
            assert len(sub_range_list) == 1
        elif ops == 'd':
            assert len(sub_range_list) == 2
        elif ops in 'uiIUD':
            assert len(sub_range_list) > 1
        elif ops in '()':
            return identify_range(i + 1, j - 1)
        else:
            raise NotImplementedError(f"Ops {ops} is not defined")
        return ops, sub_range_list

    _b = 0
    _e = len(fosq_formula) - 1
    todo_ranges.append((_b, _e))
    while (_b, _e) not in cached_objects:
        i, j = todo_ranges[-1]

        if (i, j) in cached_subranges:
            ops, sub_range_list = cached_subranges[(i, j)]
        else:
            ops, sub_range_list = identify_range(i, j)
            cached_subranges[(i, j)] = (ops, sub_range_list)

        valid_sub_ranges = True
        for _i, _j in sub_range_list:
            if not (_i, _j) in cached_objects:
                todo_ranges.append((_i, _j))
                valid_sub_ranges = False

        if valid_sub_ranges is True:
            sub_objects = [cached_objects[r] for r in sub_range_list]
            obj = ops_dict[ops](*sub_objects)
            todo_ranges.pop(-1)
            cached_objects[(i, j)] = obj
    return cached_objects[_b, _e]


op_candidates_dict = {
    "p": "epiu",
    "n": "epiu",
    "i": {1: "epiu", 2: "epiun"},
    "u": {1: "epiu", 2: "epiu"}
}


def binary_formula_iterator(depth=5,
                            num_anchor_nodes=4,
                            op_candidates=None, negation_length=True):
    # decide the ops, we didn't consider the negation as the top-level operator
    if op_candidates is None:
        op_candidates = "epiu"

    # when the depth is 1, we have only "e" to choose
    if depth == 1:
        op_candidates = "e"

    for op in op_candidates:
        if (op == 'e' and num_anchor_nodes == 1):
            yield "(p,(e))"
        elif op in 'np':
            if negation_length:
                arg_candidate_iterator = binary_formula_iterator(
                    depth=depth - 1,
                    num_anchor_nodes=num_anchor_nodes,
                    op_candidates=op_candidates_dict[op], negation_length=negation_length)
                for f in arg_candidate_iterator:
                    yield f"({op},{f})"
            else:
                pass
        elif op in 'iu':
            for arg1_num_anchor_nodes in range(1, num_anchor_nodes):
                arg2_num_anchor_nodes = num_anchor_nodes \
                                        - arg1_num_anchor_nodes
                arg1_candidate_iterator = binary_formula_iterator(
                    depth=depth,
                    num_anchor_nodes=arg1_num_anchor_nodes,
                    op_candidates=op_candidates_dict[op][1],
                    negation_length=negation_length
                )
                arg2_candidate_iterator = binary_formula_iterator(
                    depth=depth,
                    num_anchor_nodes=arg2_num_anchor_nodes,
                    op_candidates=op_candidates_dict[op][2],
                    negation_length=negation_length
                )
                for f1, f2 in product(arg1_candidate_iterator,
                                      arg2_candidate_iterator):
                    yield f"({op},{f1},{f2})"


def copy_query(q: FirstOrderSetQuery, deep=False) -> FirstOrderSetQuery:
    op = q.__o__
    if op == 'e':
        _q = Entity()
        _q.entities = q.entities
        return _q
    elif op == 'p':
        _q = Projection()
        _q.relations = q.relations
        if deep:
            _q.query = copy_query(q.query, deep)
        return _q
    elif op == 'n':
        _q = Negation()
        if deep:
            _q.query = copy_query(q.query, deep)
        return _q
    elif op in 'uiD':
        _q = ops_dict[op]()
        if deep:
            _q.sub_queries = [copy_query(sq, deep) for sq in q.sub_queries]
        return _q
    elif op == 'd':
        _q = Difference(lq=copy_query(q.lquery, deep), rq=copy_query(q.rquery, deep))
        return _q
    else:
        raise NotImplementedError


def projection_sink(fosq: FirstOrderSetQuery,
                    upper_projection_stack=[]) -> FirstOrderSetQuery:
    """Move the projections at the bottom of the tree, i.e.,
    we only allow p -> p/e
    """
    _upper_projection_stack = [copy_query(p)
                               for p in upper_projection_stack]
    if fosq.__o__ == 'p':
        _upper_projection_stack += [copy_query(fosq)]
        return projection_sink(fosq.query, _upper_projection_stack)
    elif fosq.__o__ == 'e':
        while len(_upper_projection_stack) > 0:
            p = _upper_projection_stack.pop(-1)
            p.query = fosq
            fosq = p
        return fosq
    elif fosq.__o__ == 'n':
        fosq.query = projection_sink(fosq.query, _upper_projection_stack)
        return fosq
    elif fosq.__o__ in 'iu':  # the inter section and union
        fosq.sub_queries = [projection_sink(q, _upper_projection_stack)
                            for q in fosq.sub_queries]
        return fosq


def negation_sink(fosq: FirstOrderSetQuery) -> FirstOrderSetQuery:
    """ Move the negation down of itersection and union. (negation sink)
        n -> i -> fosq should be converted to u -> n -> fosq
        n -> u -> fosq should be converted to i -> n -> fosq
    """
    if fosq.__o__ == 'e':
        return fosq
    elif fosq.__o__ == 'p':
        return fosq
    elif fosq.__o__ == 'n':
        sub_q = fosq.query
        # de Morgan rule 1
        if sub_q.__o__ == 'i':
            sub_sub_qs = sub_q.sub_queries
            _fosq = Union(
                *[Negation(q=negation_sink(q)) for q in sub_sub_qs]
            )
        elif sub_q.__o__ == 'u':
            sub_sub_qs = sub_q.sub_queries
            _fosq = Intersection(
                *[Negation(q=negation_sink(q)) for q in sub_sub_qs]
            )
        else:
            _fosq = fosq
        return _fosq

    elif fosq.__o__ in 'iu':
        fosq.sub_queries = [negation_sink(q) for q in fosq.sub_queries]
        return fosq
    else:
        raise NotImplementedError


def concate_n_chains(query: FirstOrderSetQuery) -> FirstOrderSetQuery:
    if query.__o__ == 'n':
        sub_query = query.query
        if sub_query.__o__ == 'n':
            query = sub_query.query
        query.query = concate_n_chains(query.query)
        return query
    elif query.__o__ in 'ui':
        query.sub_queries = [concate_n_chains(q) for q in query.sub_queries]
        return query
    elif query.__o__ == 'e':
        return query
    elif query.__o__ == 'p':
        query.query = concate_n_chains(query.query)
        return query
    else:
        raise NotImplementedError


def DeMorgan_replacement(query):
    """
    Input query system epiun
    Replace the u by n-i-n
    """
    if query.__o__ == 'u':
        sub_queries = [DeMorgan_replacement(q) for q in query.sub_queries]
        negated_sub_queries = [Negation(q) for q in sub_queries]
        inter = Intersection(*negated_sub_queries)
        out = Negation(inter)
        return out
    elif query.__o__ == 'i':
        sub_queries = [DeMorgan_replacement(q) for q in query.sub_queries]
        query.sub_queries = sub_queries
        return query
    elif query.__o__ == 'e':
        return query
    else:  # n and p case
        subq = query.query
        query.query = DeMorgan_replacement(subq)
        return query


def intersection_bubble(fosq: FirstOrderSetQuery) -> FirstOrderSetQuery:
    pass


def union_bubble(fosq: FirstOrderSetQuery) -> FirstOrderSetQuery:
    """ Move the union at the top of the tree
    For any i -> u pairs, we will make it as a u -> i pair
    If we use projection sink, de Morgan rule and union bubble, then we get dnf
    We handle the situation where (A or B) and (C), it should be (A and C) or (B and C)
    """
    if fosq.__o__ == 'e':
        return fosq
    elif fosq.__o__ == 'n':
        fosq.query = union_bubble(fosq.query)
        return fosq
    elif fosq.__o__ == 'p':
        sub_query = fosq.query
        if sub_query.__o__ == 'u':
            # the projection should be applid to those queries
            sub_queries = []
            for ssq in sub_query.sub_queries:
                p = copy_query(fosq)
                p.query = ssq
                sub_queries.append(p)
            return Union(*sub_queries)
        else:
            fosq.query = union_bubble(sub_query)
            return fosq
    elif fosq.__o__ == 'i':
        fosq.sub_queries = [union_bubble(q) for q in fosq.sub_queries]

        union_subq = None
        other_subq = []
        for q in fosq.sub_queries:
            if q.__o__ == 'u' and union_subq is None:
                union_subq = q
            else:
                other_subq.append(q)
        if union_subq is None:
            return fosq

        if len(other_subq) == 1:
            C = other_subq[0]
        else:
            C = Intersection(*other_subq)
        _fosq = Union(
            *[Intersection(q, copy_query(C, deep=True))
              for q in union_subq.sub_queries]
        )
        return union_bubble(_fosq)
    elif fosq.__o__ == 'u':
        fosq.sub_queries = [union_bubble(q) for q in fosq.sub_queries]
        return fosq
    else:
        raise NotImplementedError


def concate_iu_chains(fosq: FirstOrderSetQuery) -> FirstOrderSetQuery:
    if fosq.__o__ in 'pn':
        fosq.query = concate_iu_chains(fosq.query)
        return fosq
    if fosq.__o__ == 'e':
        return fosq
    if fosq.__o__ in 'iu':
        op = fosq.__o__
        same_root_queries = []
        other_queries = []
        for q in fosq.sub_queries:
            if q.__o__ == op:
                same_root_queries.append(q)
            else:
                other_queries.append(q)
        if len(same_root_queries) == 0:
            fosq.sub_queries = [concate_iu_chains(q) for q in fosq.sub_queries]
            return fosq
        sub_queries = other_queries
        for q in same_root_queries:
            sub_queries += q.sub_queries
        _fosq = ops_dict[op](*sub_queries)
        assert _fosq.formula != fosq.formula
        return concate_iu_chains(_fosq)
    if fosq.__o__ == 'd':
        fosq.lquery = concate_iu_chains(fosq.lquery)
        fosq.rquery = concate_iu_chains(fosq.rquery)
        return fosq
    if fosq.__o__ in 'DIU':
        fosq.sub_queries = [concate_iu_chains(q) for q in fosq.sub_queries]
        return fosq


def to_D(fosq):
    if fosq.__o__ == "i":
        negated = []
        not_negated = []
        for subq in fosq.sub_queries:
            if subq.__o__ == 'n':
                negated.append(to_D(subq))
            else:
                not_negated.append(to_D(subq))
        if len(negated) == 0:
            fosq.sub_queries = not_negated
            return fosq
        if len(not_negated) > 1:
            first_query = Intersection(*not_negated)
        else:
            first_query = not_negated[0]
        rest_query = [q.query for q in negated]
        multi_diff_query = [first_query] + rest_query
        fosq = Multiple_Difference(*multi_diff_query)
        return fosq
    elif fosq.__o__ == "u":
        sub_queries = [to_D(q) for q in fosq.sub_queries]
        fosq.sub_queries = sub_queries
        return fosq
    elif fosq.__o__ in 'pn':
        fosq.query = to_D(fosq.query)
        return fosq
    elif fosq.__o__ == 'e':
        return fosq
    else:
        raise NotImplementedError


def decompose_D(fosq):
    if fosq.__o__ == 'D':
        first, *rest = fosq.sub_queries
        while len(rest) > 1:
            first = Multiple_Difference(*[first, rest[0]])
            rest = rest[1:]
        fosq.sub_queries = [first, rest[0]]
        return fosq
    elif fosq.__o__ in 'iuIU':
        fosq.sub_queries = [decompose_D(q) for q in fosq.sub_queries]
        return fosq
    elif fosq.__o__ in 'pn':
        fosq.query = decompose_D(fosq.query)
        return fosq
    elif fosq.__o__ == 'e':
        return fosq


def to_d(query):
    """
    Convert the i-n like difference into difference,
    all binary operators
    """
    if query.__o__ == "i":
        assert len(query.sub_queries) == 2
        q1, q2 = query.sub_queries
        q1, q2 = to_d(q1), to_d(q2)

        if q1.__o__ == 'n' and q2.__o__ != 'n':
            return Difference(lq=q2, rq=q1.query)
        elif q1.__o__ != 'n' and q2.__o__ == 'n':
            return Difference(lq=q1, rq=q2.query)
        else:
            query.sub_queries = [q1, q2]
            return query
    elif query.__o__ == "u":
        sub_queries = [to_d(q) for q in query.sub_queries]
        query.sub_queries = sub_queries
        return query
    elif query.__o__ in 'pn':
        query.query = to_d(query.query)
        return query
    elif query.__o__ == 'e':
        return query
    else:
        raise NotImplementedError


def transformation(query, trans_func):
    original_formula = query.formula
    query = trans_func(query)
    while query.formula != original_formula:
        original_formula = query.formula
        query = trans_func(query)
    return query


def count_query_depth(query):
    if query.__o__ == 'e':
        return 0
    elif query.__o__ in 'uiUID':
        return max(count_query_depth(q) for q in query.sub_queries)
    elif query.__o__ == 'p':
        return count_query_depth(query.query) + 1
    elif query.__o__ == 'n':
        return count_query_depth(query.query)
    elif query.__o__ == 'd':
        return max(count_query_depth(query.lquery), count_query_depth(query.rquery))
    else:
        raise NotImplementedError


def count_depth(formula):
    try:
        query = parse_formula(formula)
    except:
        print(formula)
    return count_query_depth(query)


def count_entities(formula):
    return Counter(formula)['e']


def count_projections(formula):
    return Counter(formula)['p']
