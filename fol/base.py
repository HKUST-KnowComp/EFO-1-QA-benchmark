from abc import ABC


class Formula(ABC):
    def __init__(self):
        pass


class Variable(Formula):
    def __init__(self):
        super().__init__()
        print("claim a variable")
        pass


class Conjunction(Formula):
    def __init__(self, lf, rf):
        super().__init__()
        print("claim a conjunction")
        self.lf, self.rf = lf, rf


class Disjunction(Formula):
    def __init__(self, lf, rf):
        super().__init__()
        print("claim a disjunction")
        self.lf, self.rf = lf, rf


class Negation(Formula):
    def __init__(self, f):
        super().__init__()
        print("claim a negation")
        self.f = f


class Projection(Formula):
    def __init__(self, f):
        super().__init__()
        print("claim a projection")
        self.f = f


# This dictionary is important, since it determines the class you use.
grammar_class = {
    'delim': '()',
    'zop': Variable,
    'uop': {'!': Negation, 'p': Projection},
    'biop': {'&': Conjunction, '|': Disjunction}
}


def get_grammar_class(gc):
    delim = gc['delim']
    zop = gc['zop']
    uop = gc['uop']
    biop = gc['biop']
    return delim, zop, uop, biop


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


def parse_beta_like(query, gc):
    delim, zop, uop, biop = get_grammar_class(gc)

    if query == 'e':
        return zop()

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
                    return parse_beta_like(last_dilim_query, grammar_class)

        elif c in biop:  # handle the conjunction and disjunction
            if len(pstack) == 0:  # only when at the top of the syntax tree
                biop_triggers.append([i, c])

        elif c in uop:  # handle the negation and projection
            if len(pstack) == 0:  # only when at the top of the syntax tree
                uop_triggers.append([i, c])

    for i, c in biop_triggers:
        lf, rf = parse_beta_like(query[:i], grammar_class), parse_beta_like(query[i + 1:], grammar_class)
        return biop[c](lf, rf)
    for i, c in uop_triggers:
        f = parse_beta_like(query[i + 1:], grammar_class)
        return uop[c](f)

    raise SyntaxError(f"Query {query} fall out of branches")


if __name__ == '__main__':
    for name in beta_query:
        case = beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f = parse_beta_like(case, grammar_class)
        print()
