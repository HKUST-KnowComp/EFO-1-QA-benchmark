from abc import ABC

class Formula(ABC):
    def __init__(self):
        pass

class Variable(Formula):
    def __init__(self):
        pass

class Conjunction(Formula):
    def __init__(self, lf, rf):
        pass

class Disjunction(Formula):
    def __init__(self, lf, rf):
        pass

