import collections
import copy
import os
import pickle
import random
from abc import ABC, abstractmethod

from fol.base import Variable, Conjunction, Disjunction, Negation, Projection
from fol.base import parse_beta_like


class Sampler(ABC):
    def __init__(self, *args, **kwargs):
        self.objects = set()
        self.easy_objects = set()

    @abstractmethod
    def sample(self):
        """A method for sampling data and handling the set operation
        objects is for the whole answers for the query, easy_objects is for the easy answers
        """
        return self.objects, self.easy_objects

    @abstractmethod
    def reverse_sample(self):
        """A method for reverse_sampling data and handling the set operation
        objects is for the whole answers for the query, easy_objects is for the easy answers
        """
        return self.objects, self.easy_objects

    @abstractmethod
    def dumps(self):
        """A method for dump the contained information into the string
        """
        return ""

    @abstractmethod
    def clear(self):
        self.objects = set()
        self.easy_objects = set()


class VariableSampler(Variable, Sampler):
    def __init__(self, candidate_entities):
        super().__init__()
        self.candidate_entities = candidate_entities

    def sample(self):
        """
        TODO
        - Sample an entity from the candidate_entities
        - Fill that in the self.objects
        - Return the sampled results
        """
        self.objects = set()
        self.easy_objects = set()
        new_variable = random.sample(self.candidate_entities, 1)[0]
        self.objects.add(new_variable)
        self.easy_objects.add(new_variable)
        return self.objects, self.easy_objects

    def reverse_sample(self, need_to_contain: bool = None, essential_point=None):
        if not need_to_contain:
            essential_point = random.sample(self.candidate_entities - set(essential_point), 1)[0]
        if essential_point is None:
            essential_point = random.sample(self.candidate_entities, 1)[0]
        return essential_point

    def dumps(self):
        obj_cat_str = ', '.join([str(obj) for obj in list(self.objects)])
        return f'{obj_cat_str}'

    def clear(self):
        self.objects = set()
        self.easy_objects = set()


class ConjunctionSampler(Conjunction, Sampler):
    def __init__(self, lf: Sampler, rf: Sampler, candidate_entities):
        super().__init__(lf, rf)
        self.candidate_entities = candidate_entities

    def sample(self):
        lobjs, lobjs_easy = self.lf.sample()
        robjs, robjs_easy = self.rf.sample()
        self.objects = lobjs.intersection(robjs)
        self.easy_objects = lobjs_easy.intersection(robjs_easy)
        return self.objects, self.easy_objects

    def reverse_sample(self, need_to_contain: bool = None, essential_point: int = None):  # must contain the same point
        if essential_point is None:
            essential_point = random.sample(self.candidate_entities, 1)[0]
        if need_to_contain is False:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lq = self.lf.reverse_sample(need_to_contain=False, essential_point=essential_point)
                rq = self.rf.reverse_sample()
            else:
                lq = self.lf.reverse_sample()
                rq = self.rf.reverse_sample(need_to_contain=False, essential_point=essential_point)
        else:  # By default, choose a common start point.
            lq = self.lf.reverse_sample(need_to_contain=True, essential_point=essential_point)
            rq = self.rf.reverse_sample(need_to_contain=True, essential_point=essential_point)
        return f'({lq})&({rq})'

    def dumps(self):
        return f'({self.lf.dumps()})&({self.rf.dumps()})'

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.lf.clear()
        self.rf.clear()


class DisjunctionSampler(Disjunction, Sampler):
    def __init__(self, lf: Sampler, rf: Sampler, candidate_entities):
        super().__init__(lf, rf)
        self.candidate_entites = candidate_entities

    def sample(self):
        lobjs, lobjs_easy = self.lf.sample()
        robjs, robjs_easy = self.rf.sample()
        self.objects = lobjs.union(robjs)
        self.easy_objects = lobjs_easy.union(robjs_easy)
        return self.objects, self.easy_objects

    def reverse_sample(self, need_to_contain: bool = None, essential_point=None):
        if essential_point is None:
            essential_point = random.sample(self.candidate_entites, 1)[0]
        choose_formula = random.randint(0, 1)
        if choose_formula == 0:
            lq = self.lf.reverse_sample(need_to_contain=True, essential_point=essential_point)
            rq = self.rf.reverse_sample()
        else:
            lq = self.lf.reverse_sample()
            rq = self.rf.reverse_sample(need_to_contain=True, essential_point=essential_point)
        return f'({lq})|({rq})'

    def dumps(self):
        return f'({self.lf.dumps()})|({self.rf.dumps()})'

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.lf.clear()
        self.rf.clear()


class NegationSamplerProto(Negation, Sampler):
    def __init__(self, f: Sampler):
        super().__init__(f)

    def sample(self):
        """ TODO
        The negation is kind of tricky, since it requires the bounds of the objects
        You may consider several ways to formulize this problem
        v1. specify the bounds
        v2. work together with the disjunction/conjunction
        v3. other ways you may propose
        """
        pass

    def reverse_sample(self):
        pass

    def dumps(self):
        return f"!({self.f.dumps()})"

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.f.clear()


class NegationSamplerV1(NegationSamplerProto, Sampler):
    def __init__(self, f: Sampler, bounds):
        super().__init__(f)
        self.bounds = bounds

    def sample(self):
        fobjs, fobjs_easy = self.f.sample()
        self.objects = self.bounds - fobjs
        self.easy_objects = self.bounds - fobjs_easy
        return self.objects, self.easy_objects

    def reverse_sample(self, need_to_contain: bool = None, essential_point: int = None):
        if need_to_contain is not None:
            need_to_contain = 1 - need_to_contain
        query = self.f.reverse_sample(need_to_contain=need_to_contain, essential_point=essential_point)
        return f"!({query})"


class NegationSamplerV2(Negation, Sampler):
    def __init__(self, f: Sampler):
        super().__init__(f)
        self.flag = True

    def sample(self):
        pass

    def reverse_sample(self):
        pass


class ProjectionSampler(Projection, Sampler):
    def __init__(self, f: Sampler, projections, projection_origin, reverse_projection, reverse_projection_origin):
        self.projections = projections  # this is a list of collections.defaultdict(set)
        self.projection_origin = projection_origin
        self.reverse_projection = reverse_projection  # reverse_projection[tail_entities][relation] = head_entities
        self.reverse_projection_origin = reverse_projection_origin
        # a possible data structure of projections:
        # projections[head_entity][relation] = tail_entities
        self.rel = None
        super().__init__(f)

    def sample(self):
        variable, variable_origin = self.f.sample()
        if len(variable) != 0:
            chosen_variable = random.sample(variable, 1)[0]
            self.rel = random.sample(self.projections[chosen_variable].keys(), 1)[0]
            self.objects = self.projections[chosen_variable][self.rel]
            self.easy_objects = self.projection_origin[chosen_variable][self.rel]
            for e in variable:
                self.objects.update(self.projections[e][self.rel])
            for e in variable_origin:
                self.easy_objects.update(self.projection_origin[e][self.rel])
            return self.objects, self.easy_objects
        else:
            return set(), set()

    def reverse_sample(self, need_to_contain: bool = None, essential_point=None):
        meeting_requirement = False  # whether have satisfied the essential_point condition
        # since the projection[next_point][self.rel]  contains essential_point even if not starting from it
        while meeting_requirement is False:
            if need_to_contain is False:
                now_point = random.sample(set(range(len(self.projections))) - {essential_point}, 1)[0]
            else:
                if essential_point is None:
                    now_point = random.sample(set(range(len(self.projections))), 1)[0]
                else:
                    now_point = essential_point
            self.rel = random.sample(self.reverse_projection[now_point].keys(), 1)[0]
            next_points = self.reverse_projection[now_point][self.rel]
            next_point = random.sample(next_points, 1)[0]
            query = self.f.reverse_sample(need_to_contain=True, essential_point=next_point)
            meeting_requirement = True
            if need_to_contain is False and now_point in self.projections[next_point][self.rel]:
                meeting_requirement = False
        return f"[{str(self.rel)}]({query})"

    def dumps(self):
        r_str = str(self.rel)  # Notice: sample first
        return f"[{r_str}]({self.f.dumps()})"

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.rel = set()
        self.f.clear()


# This section is important, since it determines the class you use.
grammar_class = {
    'delim': '()',
    'zop': VariableSampler,
    'uop': {'~': NegationSamplerV1, 'p': ProjectionSampler},
    'biop': {'&': ConjunctionSampler, '|': DisjunctionSampler}
}


def get_grammar_class(gc):
    delim = gc['delim']
    zop = gc['zop']
    uop = gc['uop']
    biop = gc['biop']
    return delim, zop, uop, biop


def sample_beta_like(query, gc, projection, projection_origin, reverse_projection, reverse_projection_origin):
    delim, zop, uop, biop = get_grammar_class(gc)

    if query == 'e':
        return zop(set(range(len(projection))))

    pstack = []
    print()
    uop_triggers = []
    biop_triggers = []

    for i, c in enumerate(query):
        print(c, end='-')
        if c in delim:  # if there is any composition
            if c == '(':
                pstack.append(i)
            else:
                if pstack:
                    begin = pstack.pop(-1)
                    if pstack:
                        continue
                    else:
                        last_dilim_query = query[begin + 1: i]
                else:
                    raise SyntaxError(f"Query {query} is Iiligal")
                # address the only bracket case
                if begin == 0 and i == len(query) - 1:
                    return sample_beta_like(last_dilim_query, grammar_class,
                                            projection, projection_origin, reverse_projection,
                                            reverse_projection_origin)

        elif c in biop:  # handle the conjunction and disjunction
            if len(pstack) == 0:  # only when at the top of the syntax tree
                biop_triggers.append([i, c])

        elif c in uop:  # handle the negation and projection
            if len(pstack) == 0:  # only when at the top of the syntax tree
                uop_triggers.append([i, c])

    for i, c in biop_triggers:
        lf, rf = sample_beta_like(query[:i], grammar_class,
                                  projection, projection_origin, reverse_projection, reverse_projection_origin), \
                 sample_beta_like(query[i + 1:], grammar_class,
                                  projection, projection_origin, reverse_projection, reverse_projection_origin)
        return biop[c](lf, rf, set(range(len(projection))))
    for i, c in uop_triggers:
        f = sample_beta_like(query[i + 1:], grammar_class,
                             projection, projection_origin, reverse_projection, reverse_projection_origin)
        if c == 'p':
            return uop[c](f, projection, projection_origin, reverse_projection, reverse_projection_origin)
        else:
            return uop[c](f, set(range(len(projection))))

    raise SyntaxError(f"Query {query} fall out of branches")


def load_data(input_edge_file, all_entity_dict, all_relation_dict, projection_origin, reverse_projection_origin):
    projections = copy.deepcopy(projection_origin)
    reverse = copy.deepcopy(reverse_projection_origin)
    with open(input_edge_file, 'r', errors='ignore') as infile:
        for line in infile.readlines():
            e1, r, e2 = line.strip().split('\t')
            r_projection = '+' + r
            r_reverse = '-' + r
            if e1 in all_entity_dict and e2 in all_entity_dict and r_projection in all_relation_dict:
                e1, r_projection, r_reverse, e2 = all_entity_dict[e1], all_relation_dict[r_projection], \
                                                  all_relation_dict[r_reverse], all_entity_dict[e2]
                projections[e1][r_projection].add(e2)
                projections[e2][r_reverse].add(e1)
                reverse[e2][r_projection].add(e1)
                reverse[e1][r_reverse].add(e2)
            else:
                pass

    return projections, reverse


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


beta_query = {
    '1p': 'p(e)',
    '2p': 'p(p(e))',
    '3p': 'p(p(p(e)))',
    '2i': 'p(e)&p(e)',
    '3i': 'p(e)&p(e)&p(e)',
    '2in': 'p(e)&!p(e)',
    '3in': 'p(e)&p(e)&!p(e)',
    'inp': 'p(p(e)&!p(e))',
    'pni': 'p(p(e))&!p(e)',
    'ip': 'p(p(e)&p(e))',
    'pi': 'p(e)&p(p(e))',
    '2u': 'p(e)|p(e)',
    'up': 'p(p(e)|p(e))'
}
'''

'''
if __name__ == "__main__":
    stanford_data_path = '../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
    projection_none = [collections.defaultdict(set) for i in range(len(all_entity_dict))]
    reverse_proection_none = [collections.defaultdict(set) for i in range(len(all_entity_dict))]
    projection_train, reverse_projection_train = load_data('../datasets_knowledge_embedding/FB15k-237/train.txt',
                                                           all_entity_dict, all_relation_dict, projection_none,
                                                           reverse_proection_none)
    projection_valid, reverse_projection_valid = load_data('../datasets_knowledge_embedding/FB15k-237/valid.txt',
                                                           all_entity_dict, all_relation_dict, projection_train,
                                                           reverse_projection_train)
    for name in beta_query:
        case = beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f = sample_beta_like(case, grammar_class, projection_train, projection_none,
                             reverse_projection_train, reverse_proection_none)
        f_valid = sample_beta_like(case, grammar_class, projection_valid, projection_train,
                                   reverse_projection_valid, reverse_projection_train)
        a = f_valid.sample()
        b = f_valid.dumps()
        d = f_valid.reverse_sample()  # possible '([30](5689))&(([30](5689))&([322](434)))'
        print(d)
