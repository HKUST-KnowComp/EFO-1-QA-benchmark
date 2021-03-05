from abc import ABC, abstractmethod


class Formula:
    def __init__(self) -> None:
        pass

class Variable(Formula):
    def __init__(self) -> None:
        super().__init__()

class Constant(Formula):
    def __init__(self, value) -> None:
        super().__init__()
        self.value

class Conjunction(Formula):
    def __init__(self, lf: Formula, rf: Formula) -> None:
        super().__init__()
        self.lf = lf
        self.rf = rf

class Disjunction(Formula):
    def __init__(self, lf: Formula, rf: Formula) -> None:
        super().__init__()
        self.lf = lf
        self.rf = rf

class Negation(Formula):
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f

class Exist(Formula):
    def __init__(self) -> None:
        super().__init__()


class ForAll(Formula):
    def __init__(self) -> None:
        super().__init__()


class MetaFOL(ABC):
    """
    First-order logical parsing
    - operators
        - & for conjunction
        - | for disjunction
        - ! for negation
    - brackets
        - ( and )
    - meta placeholders
        - e entity
        - r projection
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass

    def parse(self, formula):
        """
        Given a sentence, we parse execute it by the first order logic
        """
        # negation of total formula
        if formula[0] == "!":
            return self.negation(self.parse(formula[1:]))

        L = len(formula)
        # identify first argument
        arg1 = None
        begin, end = -1, -1
        if formula[0] == '(':  # the first argument is a formula
            stack = []
            for i in range(L):
                if formula[i] == '(':
                    stack.append(i)
                if formula[i] == ')':
                    begin = stack.pop(-1) + 1
                    if len(stack) == 0:
                        end = i
                        arg1 = formula[begin: end]
                    break
        else:
            arg1 = formula[0]
            end = 0
        if arg1 is None:
            raise SyntaxError("Incorrect FOL formula")

        # check whether there is a arg2
        if end + 1 == L:  # if this bracket is redundent
            return self.parse(arg1)

        if formula[end + 1] in '&|': # there is a connector
            arg2 = formula[end+2:]
            if formula[end + 1] == '&':
                return self.conjunction(self.parse(arg1), self.parse(arg2))
            if formula[end + 1] == '|':
                return self.disjunction(self.parse(arg1), self.parse(arg2))
        else: # there is no connector
            return self.parse(arg1)

    def load_from(self, stanford):
        """
        load sentence from stanford parsed sentences
        """
        pass

    def generate(self, sentence):
        pass


class FOLPrinter(MetaFOL):
    def conjunction(self, a, b):
        print("conjunction: {}, {}".format(a, b))

    def disjunction(self, a, b):
        print("disjunction: {}, {}".format(a, b))

    def negation(self, a):
        print("negation: {}".format(a))

if __name__ == '__main__':
    FOLPrinter().parse("(a)&(b)")