import collections
import copy
import os
import pickle
import random
from abc import ABC, abstractmethod

from base import Variable, Conjunction, Disjunction, Negation, Projection
from base import parse_beta_like


class Reader(ABC):  # A Reader-like class, for the purpose of getting answers for a given query
    def __init__(self, *args, **kwargs):
        self.objects = set()
        self.easy_objects = set()

    @abstractmethod
    def read(self):
        """A method for getting answers for query
        objects is for the whole answers for the query, easy_objects is for the easy answers
        """
        return self.objects, self.easy_objects

    @abstractmethod
    def clear(self):
        self.objects = set()
        self.easy_objects = set()


class VariableReader(Variable, Reader):
    def __init__(self, relation, projection, projection_origin):
        super().__init__()
        self.rel = relation
        self.projection = projection
        self.projection = projection_origin

    def read(self):
        """
        TODO
        - read an entity from the candidate_entities
        - Fill that in the self.objects
        - Return the readd results
        """
        return self.objects, self.easy_objects

    def clear(self):
        self.objects = set()
        self.easy_objects = set()


class ConjunctionReader(Conjunction, Reader):
    def __init__(self, lf: Reader, rf: Reader):
        super().__init__(lf, rf)

    def read(self):
        """
        TODO
        """
        lobjs, lobjs_easy = self.lf.read()
        robjs, robjs_easy = self.rf.read()
        self.objects = lobjs.intersection(robjs)
        self.easy_objects = lobjs_easy.intersection(robjs_easy)
        return self.objects, self.easy_objects

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.lf.clear()
        self.rf.clear()


class DisjunctionReader(Disjunction, Reader):
    def __init__(self, lf: Reader, rf: Reader):
        super().__init__(lf, rf)

    def read(self):
        """
        TODO
        """
        lobjs, lobjs_easy = self.lf.read()
        robjs, robjs_easy = self.rf.read()
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


class NegationReaderProto(Negation, Reader):
    def __init__(self, f: Reader):
        super().__init__(f)

    def read(self):
        """ TODO
        The negation is kind of tricky, since it requires the bounds of the objects
        You may consider several ways to formulize this problem
        v1. specify the bounds
        v2. work together with the disjunction/conjunction
        v3. other ways you may propose
        """
        pass

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.f.clear()


class NegationReaderV1(NegationReaderProto, Reader):
    def __init__(self, f: Reader, bounds):
        super().__init__(f)
        self.bounds = bounds

    def read(self):
        fobjs, fobjs_easy = self.f.read()
        self.objects = self.bounds - fobjs
        self.easy_objects = self.bounds - fobjs_easy
        return self.objects, self.easy_objects


class NegationReaderV2(Negation, Reader):
    def __init__(self, f: Reader):
        super().__init__(f)
        self.flag = True

    def read(self):
        pass


class ProjectionReader(Projection, Reader):
    def __init__(self, f: Reader, projections, projection_origin):
        self.projections = projections  # this is abstracted from knowledge graphs  is a list of collections.defaultdict(set)
        self.projection_origin = projection_origin
        self.rel = None  # TODO to fill, the relation you select
        super().__init__(f)

    def read(self):
        '''
        Use projection and projection_origin to get all answers and easy_answers
        '''
        pass

    def clear(self):
        self.objects = set()
        self.easy_objects = set()
        self.rel = set()
        self.f.clear()


def read_beta_like(query, gc, projection,
                   projection_origin) -> Reader:  # a parse_beta_like function, will return a Reader class.
    '''
    TODO: Use the grammar logic in parse_beta_like to return a Reader class
    Use the read function in Reader class to give all answers or easy_answers of the query
    '''
    pass


grammar_class = {
    'delim': '()',
    'zop': VariableReader,
    'uop': {'!': NegationReaderV1, 'p': ProjectionReader},
    'biop': {'&': ConjunctionReader, '|': DisjunctionReader}
}