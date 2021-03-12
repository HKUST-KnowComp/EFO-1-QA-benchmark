import collections
import os
import pickle
import random
from abc import ABC, abstractmethod

from base import Variable, Conjunction, Disjunction, Negation, Projection
from base import parse_beta_like


class Sampler(ABC):
    def __init__(self, *args, **kwargs):
        self.objects = set()

    @abstractmethod
    def sample(self):
        """A method for sampling data and handling the set operation
        """
        return self.objects, None

    @abstractmethod
    def dumps(self):
        """A method for dump the contained information into the string
        """
        return ""

    @abstractmethod
    def clear(self):
        self.objects = set()


class VariableSampler(Variable, Sampler):
    def __init__(self, candidate_entities):
        super().__init__()
        self.candidate_entities = candidate_entities
        self.objects = set()

    def sample(self):
        """
        TODO
        - Sample an entity from the candidate_entities
        - Fill that in the self.objects
        - Return the sampled results
        """
        new_variable = random.sample(self.candidate_entities, 1)[0]
        self.objects.add(new_variable)
        return self.objects

    def dumps(self):
        obj_cat_str = ', '.join([str(obj) for obj in list(self.objects)])
        return f'{obj_cat_str}'

    def clear(self):
        self.objects = set()


class ConjunctionSampler(Conjunction, Sampler):
    def __init__(self, lf: Sampler, rf: Sampler):
        super().__init__(lf, rf)

    def sample(self):
        """
        TODO
        """
        lobjs = self.lf.sample()
        robjs = self.rf.sample()
        self.objects = lobjs.intersection(robjs)
        return self.objects

    def dumps(self):
        return f'({self.lf.dumps()})&({self.rf.dumps()})'

    def clear(self):
        self.objects = set()
        self.lf.clear()
        self.rf.clear()


class DisjunctionSampler(Disjunction, Sampler):
    def __init__(self, lf: Sampler, rf: Sampler):
        super().__init__(lf, rf)

    def sample(self):
        """
        TODO
        """
        lobjs = self.lf.sample()
        robjs = self.rf.sample()
        self.objects = lobjs.union(robjs)
        return self.objects

    def dumps(self):
        return f'({self.lf.dumps()})|({self.rf.dumps()})'

    def clear(self):
        self.objects = set()
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

    def dumps(self):
        return f"!({self.f.dumps()})"

    def clear(self):
        self.objects = set()
        self.f.clear()


class NegationSamplerV1(NegationSamplerProto, Sampler):
    def __init__(self, f: Sampler, bounds):
        super().__init__(f)
        self.bounds = bounds

    def sample(self):
        fobjs = self.f.sample()
        self.objects = self.bounds - fobjs
        return self.objects


class NegationSamplerV2(Negation, Sampler):
    def __init__(self, f: Sampler):
        super().__init__(f)
        self.flag = True

    def sample(self):
        pass


class ProjectionSampler(Projection, Sampler):
    def __init__(self, f: Sampler, projections):
        self.projections = projections  # this is abstracted from knowledge graphs  is a list of collections.defaultdict
        # a possible data structure of projections:
        # projections[head_entity][relation] = tail_entities
        self.rel = None  # TODO to fill, the relation you select
        super().__init__(f)

    def sample(self):
        variable = self.f.sample()
        if len(variable) != 0:
            chosen_variable = random.sample(variable, 1)[0]
            self.rel = random.sample(projection[chosen_variable].keys(), 1)[0]
            self.objects = projection[chosen_variable][self.rel]
            for e in variable:
                self.objects.update(projection[e][self.rel])
            return self.objects
        else:
            return set()

    def dumps(self):
        r_str = str(self.rel)  # Notice: sample first
        return f"[{r_str}]({self.f.dumps()})"

    def clear(self):
        self.objects = set()
        self.rel = set()
        self.f.clear()


# This section is important, since it determines the class you use.
grammar_class = {
    'delim': '()',
    'zop': VariableSampler,
    'uop': {'!': NegationSamplerV1, 'p': ProjectionSampler},
    'biop': {'&': ConjunctionSampler, '|': DisjunctionSampler}
}


def get_grammar_class(gc):
    delim = gc['delim']
    zop = gc['zop']
    uop = gc['uop']
    biop = gc['biop']
    return delim, zop, uop, biop


def sample_beta_like(query, gc, projection):
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
                    return sample_beta_like(last_dilim_query, grammar_class, projection)

        elif c in biop:  # handle the conjunction and disjunction
            if len(pstack) == 0:  # only when at the top of the syntax tree
                biop_triggers.append([i, c])

        elif c in uop:  # handle the negation and projection
            if len(pstack) == 0:  # only when at the top of the syntax tree
                uop_triggers.append([i, c])

    for i, c in biop_triggers:
        lf, rf = sample_beta_like(query[:i], grammar_class, projection), sample_beta_like(query[i + 1:], grammar_class,
                                                                                          projection)
        return biop[c](lf, rf)
    for i, c in uop_triggers:
        f = sample_beta_like(query[i + 1:], grammar_class, projection)
        if c == 'p':
            return uop[c](f, projection)
        else:
            return uop[c](f, set(range(len(projection))))

    raise SyntaxError(f"Query {query} fall out of branches")


def load_data(input_edge_file, all_entity_dict, all_relation_dict):
    n = len(all_entity_dict)
    projections = [collections.defaultdict(set) for i in range(n)]
    with open(input_edge_file, 'r', errors='ignore') as infile:
        for line in infile.readlines():
            e1, r, e2 = line.strip().split('\t')
            r_projection = '+' + r
            r_reverse = '-' + r
            e1, r_projection, r_reverse, e2 = all_entity_dict[e1], all_relation_dict[r_projection], \
                                              all_relation_dict[r_reverse], all_entity_dict[e2]
            projections[e1][r_projection].add(e2)
            projections[e2][r_reverse].add(e1)
    return projections


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

if __name__ == "__main__":
    """You can use the function `parse_beta_like(query, gc)`
    """
    stanford_data_path = '../data/FB15k-237-betae'
    all_entity_dict, all_relation_dict, id2ent, id2rel = read_indexing(stanford_data_path)
    projection = load_data('../datasets_knowledge_embedding/FB15k-237/train.txt', all_entity_dict, all_relation_dict)
    for name in beta_query:
        case = beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f = sample_beta_like(case, grammar_class, projection)
        a = f.sample()
        b = f.dumps()
        print(a, b)
        print()
