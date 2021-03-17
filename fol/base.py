from abc import ABC
import random
from collections import defaultdict
class Formula(ABC):
    def __init__(self):
        pass


class Variable(Formula):
    def __init__(self):
        super().__init__()
        pass


class Conjunction(Formula):
    def __init__(self, lf, rf):
        super().__init__()
        self.lf, self.rf = lf, rf


class Disjunction(Formula):
    def __init__(self, lf, rf):
        super().__init__()
        self.lf, self.rf = lf, rf


class Negation(Formula):
    def __init__(self, f):
        super().__init__()
        self.f = f


class Projection(Formula):
    def __init__(self, f):
        super().__init__()
        self.f = f

class Difference(Formula):
    def __init__(self, lf, rf):
        super().__init__()
        self.lf = lf
        self.rf = rf

# This dictionary is important, since it determines the class you use.
grammar_class = {
    'delim': '()',
    'zop': Variable,
    'uop': {'~': Negation, 'p': Projection},
    'biop': {'&': Conjunction, '|': Disjunction, '-': Difference}
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

def generate_meta_query(d=0, max_depth=5):
    if d > max_depth:
        return "e"

    t = random.randint(0, 4)
    if t == 0:
        return f"p({generate_meta_query(d+1, max_depth)})"
    if t == 1:
        return f"~({generate_meta_query(d+1, max_depth)})"
    if t == 2:
        return f"({generate_meta_query(d+1, max_depth)})&({generate_meta_query(d+1, max_depth)})"
    if t == 3:
        return f"({generate_meta_query(d+1, max_depth)})|({generate_meta_query(d+1, max_depth)})"
    if t == 4:
        return f"({generate_meta_query(d+1, max_depth)})-({generate_meta_query(d+1, max_depth)})"


def parse_meta_query(query, gc):
    delim, zop, uop, biop = get_grammar_class(gc)

    if query == 'e':
        return zop()

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
                    return parse_meta_query(last_dilim_query, grammar_class)

        elif c in biop:  # handle the conjunction and disjunction
            if len(pstack) == 0:  # only when at the top of the syntax tree
                biop_triggers.append([i, c])

        elif c in uop:  # handle the negation and projection
            if len(pstack) == 0:  # only when at the top of the syntax tree
                uop_triggers.append([i, c])

    for i, c in biop_triggers:
        lf, rf = parse_meta_query(query[:i], grammar_class), parse_meta_query(query[i + 1:], grammar_class)
        return biop[c](lf, rf)
    for i, c in uop_triggers:
        f = parse_meta_query(query[i + 1:], grammar_class)
        return uop[c](f)

    raise SyntaxError(f"Query {query} fall out of branches")


if __name__ == '__main__':
    # test beta type query
    for name in beta_query:
        case = beta_query[name]
        print(f'parsing the query {name}: `{case}`')
        f = parse_meta_query(case, grammar_class)
        print()

    # test random generate query
    data = defaultdict(list)
    for i in range(100):
        for j in range(10):
            meta_q = generate_meta_query()
            parse_meta_query(meta_q, grammar_class)
            data['id'].append(i)
            data['depth'].append(j)
            data['meta_query'].append(meta_q)
    import pandas as pd
    df = pd.DataFrame(data=data)
    df.to_csv('fol/random_meta_query.csv', index=False)

