import collections
import copy
import os
import pickle
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from fol.base import Variable, Conjunction, Disjunction, Negation, Projection, Difference


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
        if need_to_contain is False:
            essential_point = random.sample(self.candidate_entities - {essential_point}, 1)[0]
        if need_to_contain is None and essential_point is None:
            essential_point = random.sample(self.candidate_entities, 1)[0]
        self.objects = {essential_point}
        self.easy_objects = {essential_point}
        return self.objects, self.easy_objects

    def dumps(self):
        obj_cat_str = ', '.join([str(obj) for obj in list(self.objects)])
        return f'{{{obj_cat_str}}}'

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
                lobjs, lobjs_easy = self.lf.reverse_sample(need_to_contain=False, essential_point=essential_point)
                robjs, robjs_easy = self.rf.reverse_sample()
            else:
                lobjs, lobjs_easy = self.lf.reverse_sample()
                robjs, robjs_easy = self.rf.reverse_sample(need_to_contain=False, essential_point=essential_point)
        else:  # By default, choose a common start point.
            lobjs, lobjs_easy = self.lf.reverse_sample(need_to_contain=True, essential_point=essential_point)
            robjs, robjs_easy = self.rf.reverse_sample(need_to_contain=True, essential_point=essential_point)
        self.objects = lobjs.intersection(robjs)
        self.easy_objects = lobjs_easy.intersection(robjs_easy)
        return self.objects, self.easy_objects

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
        if need_to_contain is False:
            lobjs, lobjs_easy = self.lf.reverse_sample(need_to_contain=False, essential_point=essential_point)
            robjs, robjs_easy = self.rf.reverse_sample(need_to_contain=False, essential_point=essential_point)
        else:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lobjs, lobjs_easy = self.lf.reverse_sample(need_to_contain=True, essential_point=essential_point)
                robjs, robjs_easy = self.rf.reverse_sample()
            else:
                lobjs, lobjs_easy = self.lf.reverse_sample()
                robjs, robjs_easy = self.rf.reverse_sample(need_to_contain=True, essential_point=essential_point)
        self.objects = lobjs.union(robjs)
        self.easy_objects = lobjs_easy.union(robjs_easy)
        return self.objects, self.easy_objects

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
            need_to_contain = bool(1 - need_to_contain)
        fobjs, fobjs_easy = self.f.reverse_sample(need_to_contain=need_to_contain, essential_point=essential_point)
        self.objects = self.bounds - fobjs
        self.easy_objects = self.bounds - fobjs_easy
        return self.objects, self.easy_objects


class DifferenceSampler(Difference, Sampler):
    def __init__(self, lf, rf, candidate_entities):
        super().__init__(lf, rf)
        self.lf = lf
        self.rf = rf
        self.candidate_entities = candidate_entities

    def sample(self):
        lfobjs, lfobjs_easy = self.lf.sample()
        rfobjs, rfobjs_easy = self.rf.sample()
        self.objects = lfobjs - rfobjs
        self.easy_objects = lfobjs_easy - rfobjs_easy
        return self.objects, self.easy_objects

    def reverse_sample(self, need_to_contain: bool = None, essential_point: int = None):
        if need_to_contain is False:
            choose_formula = random.randint(0, 1)
            if choose_formula == 0:
                lfobjs, lfobjs_easy = self.lf.reverse_sample(need_to_contain=False, essential_point=essential_point)
                rfobjs, rfobjs_easy = self.rf.reverse_sample()
            else:
                lfobjs, lfobjs_easy = self.lf.reverse_sample()
                rfobjs, rfobjs_easy = self.rf.reverse_sample(need_to_contain=True, essential_point=essential_point)
        else:
            lfobjs, lfobjs_easy = self.lf.reverse_sample(need_to_contain=True, essential_point=essential_point)
            rfobjs, rfobjs_easy = self.rf.reverse_sample(need_to_contain=False, essential_point=essential_point)
        self.objects = lfobjs - rfobjs
        self.easy_objects = lfobjs_easy - rfobjs_easy
        return self.objects, self.easy_objects

    def dumps(self):
        return f'({self.lf.dumps()})-({self.rf.dumps()})'

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.lf.clear()
        self.rf.clear()


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
            meeting_requirement = True
            if need_to_contain is False and essential_point in self.projections[next_point][self.rel]:
                meeting_requirement = False
        f_object, f_object_easy = self.f.reverse_sample(need_to_contain=True, essential_point=next_point)
        if None in f_object:
            raise ValueError
        self.objects = self.projections[next_point][self.rel]
        self.easy_objects = self.projection_origin[next_point][self.rel]
        for entity in f_object.copy():
            if type(entity) == int:
                self.objects.update(self.projections[entity][self.rel])
        for entity in f_object.copy():
            if type(entity) == int:
                self.easy_objects.update(self.projection_origin[entity][self.rel])
        return self.objects, self.easy_objects

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
    'biop': {'&': ConjunctionSampler, '|': DisjunctionSampler, '-': DifferenceSampler}
}


def get_grammar_class(gc):
    delim = gc['delim']
    zop = gc['zop']
    uop = gc['uop']
    biop = gc['biop']
    return delim, zop, uop, biop


def parse_string(query, gc, projection, projection_origin, reverse_projection, reverse_projection_origin):
    delim, zop, uop, biop = get_grammar_class(gc)

    if query == 'e' or query[0] == '{':
        if query == 'e':
            return zop(set(range(len(projection))))
        else:
            i = 1
            while query[i] != '}':
                i += 1
            entity = int(query[1:i])
            return zop({entity})
    pstack = []
    uop_triggers = []
    biop_triggers = []

    for i, c in enumerate(query):
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
                    return parse_string(last_dilim_query, grammar_class,
                                        projection, projection_origin, reverse_projection,
                                        reverse_projection_origin)

        elif c in biop:  # handle the conjunction and disjunction
            if len(pstack) == 0:  # only when at the top of the syntax tree
                biop_triggers.append([i, c])

        elif c in uop:  # handle the negation and projection
            if len(pstack) == 0:  # only when at the top of the syntax tree
                uop_triggers.append([i, c])

    for i, c in biop_triggers:
        lf, rf = parse_string(query[:i], grammar_class,
                              projection, projection_origin, reverse_projection, reverse_projection_origin), \
                 parse_string(query[i + 1:], grammar_class,
                              projection, projection_origin, reverse_projection, reverse_projection_origin)
        return biop[c](lf, rf, set(range(len(projection))))
    for i, c in uop_triggers:
        f = parse_string(query[i + 1:], grammar_class,
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


def compare_depth_query(depth1, depth2, depth_dict,
                        projection_hard, projection_origin, reverse_hard, reverse_origin, grammer_class,
                        start_point_num, query_num):
    start_point_set = random.sample(set(range(len(projection_hard))), start_point_num)
    print(start_point_set)
    stored_similarity = collections.defaultdict(dict)
    for ms1 in depth_dict[depth1][:query_num]:
        for ms2 in depth_dict[depth2][:query_num]:
            stored_similarity[ms1][ms2] = []
            sampler1 = parse_string(ms1, grammar_class, projection_hard, projection_origin, reverse_hard,
                                    reverse_origin)
            sampler2 = parse_string(ms2, grammar_class, projection_hard, projection_origin, reverse_hard,
                                    reverse_origin)
            for start_point in start_point_set:
                sampled_ans1, sampled_ans1_easy = sampler1.reverse_sample(need_to_contain=True,
                                                                          essential_point=start_point)
                sampled_ans2, sampled_ans2_easy = sampler2.reverse_sample(need_to_contain=True,
                                                                          essential_point=start_point)
                all_ans = sampled_ans1.union(sampled_ans2)
                shared_ans = sampled_ans1.intersection(sampled_ans2)
                similarity = len(shared_ans) / len(all_ans)
                stored_similarity[ms1][ms2].append(similarity)
    return stored_similarity


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

partial_beta_query = {
    '1p': 'p({3})',
    '2p': 'p(p({4}))',
    '3p': 'p(p(p({7})))',
    '2i': 'p(e)&p({1003})',
    '3i': 'p(e)&p({5133})&p(e)',
    '2in': 'p(e)&!p({876})',
    '3in': 'p({971})&p(e)&!p(e)',
    'inp': 'p(p(e)&!p({765}))',
    'pni': 'p(p({10074}))&!p(e)',
    'ip': 'p(p(e)&p(e))',
    'pi': 'p(e)&p(p({1087}))',
    '2u': 'p(e)|p({7087})',
    'up': 'p(p(e)|p({21}))'
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
    '''                                                   
    for name in beta_query:
        case = beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f = parse_string(case, grammar_class, projection_train, projection_none,
                             reverse_projection_train, reverse_proection_none)
        f_valid = parse_string(case, grammar_class, projection_valid, projection_train,
                                   reverse_projection_valid, reverse_projection_train)
        a = f_valid.sample()
        b = f_valid.dumps()
        d_ans, d_easy_ans = f_valid.reverse_sample(essential_point=2)
        e = f_valid.dumps()
        print(e)
    '''
    '''
    for name in partial_beta_query:
        case = partial_beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f_valid = parse_string(case, grammar_class, projection_valid, projection_train,
                               reverse_projection_valid, reverse_projection_train)
        e, e_easy = f_valid.sample()
        g = f_valid.dumps()
        h, h_easy = f_valid.reverse_sample()
        i = f_valid.dumps()
        print(g, e, e_easy)
        print(i, h, h_easy)
    '''
    import pandas as pd

    generated_meta_string = pd.read_csv('random_meta_query.csv').to_dict()
    all_depth_meta_string = collections.defaultdict(list)
    for idx in generated_meta_string['meta_query']:
        meta_string = generated_meta_string['meta_query'][idx]
        depth = generated_meta_string['depth'][idx]
        all_depth_meta_string[depth].append(meta_string)
    all_similarity = compare_depth_query(depth1=3, depth2=9, depth_dict=all_depth_meta_string,
                                         projection_hard=projection_train,
                                         projection_origin=projection_none, reverse_hard=reverse_projection_train,
                                         reverse_origin=reverse_proection_none, grammer_class=grammar_class,
                                         start_point_num=10, query_num=10)
    data = np.zeros(1000, dtype=float)
    idx = 0
    for ms1 in all_similarity.keys():
        for ms2 in all_similarity[ms1].keys():
            for s in all_similarity[ms1][ms2]:
                data[idx] = s
                idx += 1
    plt.hist(data, bins=40, density=True)
    plt.show()
    with open('similarity.txt', 'w') as outfile:
        json.dump(all_similarity, outfile)

