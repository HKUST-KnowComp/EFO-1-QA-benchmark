from abc import ABC, abstractmethod
from .base import Variable, Conjunction, Disjunction, Negation, Projection
from .base import parse_beta_like

class Sampler(ABC):
    def __init__(self, *args, **kwargs):
        self.objects = set()
        
    @abstractmethod
    def sample(self):
        return self.objects
    

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
        self.objects.add(1)
        return self.objects


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

class NegationSampler_Proto(Negation, Sampler):
    def __init__(self, f):
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

class NegationSamplerV1(Negation, Sampler):
    def __init__(self, f):
        super().__init__(f)
    
    def sample(self):
        pass

class NegationSamplerV2(Negation, Sampler):
    def __init__(self, f):
        super().__init__(f)
        
    def sample(self):
        pass

class ProjectionSampler(Projection, Sampler):
    def __init__(self, f, projections):
        def __init__(self, f):
            super().__init__(f)
        
        self.projections = projections # this is abstracted from knowledge graphs
        # a possible data structure of projections:
        # projections[head_entity][relation] = tail_entities

    def sample(self):
        pass


# This section is important, since it determines the class you use.
grammar_class = {
    'delim': '()',
    'zop': VariableSampler,
    'uop': {'!': NegationSamplerV1, 'p': ProjectionSampler},
    'biop': {'&': ConjunctionSampler, '|': DisjunctionSampler}
}


if __name__ == "__main__":
    """You can use the function `parse_beta_like(query, gc)`
    """
    pass