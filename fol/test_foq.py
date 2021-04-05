from foq import *

if __name__ == "__main__":
    beta_query = {
        '1p': 'p(e)',
        '2p': 'p(p(e))',
        '3p': 'p(p(p(e)))',
        '2i': 'p(e)&p(e)',
        '3i': 'p(e)&p(e)&p(e)',
        '2in': 'p(e)-p(e)',
        '3in': 'p(e)&p(e)-p(e)',
        'inp': 'p(p(e)-p(e))',
        'pni': 'p(p(e))-p(e)',
        'ip': 'p(p(e)&p(e))',
        'pi': 'p(e)&p(p(e))',
        '2u': 'p(e)|p(e)',
        'up': 'p(p(e)|p(e))'
    }
    # test 1 print meta string
    print('-'*10)
    print("Test 1, parse meta formula")
    for k, v in beta_query.items():
        print(f"[Test1] parse beta query {k} with our grammar {v}")
        obj = parse_foq_formula(v)
        print(f"[Test1][nested class]{obj.meta_str}")
        print(f"[Test1][meta formula]{obj.meta_formula}")
        print(f"[Test1][ground formula]{obj.ground_formula}")
        print()

        oobj = parse_foq_formula(obj.meta_formula)
        assert oobj.meta_formula == obj.meta_formula

    print('-'*10)
    print("Test 2, parse grounded formula")
    import random
    def random_e_ground(foq_formula):
        for i, c in enumerate(foq_formula):
            if c == 'e':
                return foq_formula[:i] + "{" + str(random.randint(0, 100)) + "}" + foq_formula[i+1:]

    def random_p_ground(foq_formula):
        for i, c in enumerate(foq_formula):
            if c == 'p':
                return foq_formula[:i] + "[" + str(random.randint(100, 200)) + "]" + foq_formula[i+1:]

    for k, v in beta_query.items():
        gv = random_p_ground(random_e_ground(v))
        print(f"[Test2] {v} is grounded into {gv}")
        obj = parse_foq_formula(v)
        gobj = parse_foq_formula(gv)
        print(f"[Test2][Meta Formula] {gobj.meta_formula}")
        print(f"[Test2][Grounded Formula] {gobj.ground_formula}")
        print()

        oobj = parse_foq_formula(obj.meta_formula)
        assert gobj.meta_formula == oobj.meta_formula
        ogobj = parse_foq_formula(gobj.ground_formula)
        assert gobj.ground_formula == ogobj.ground_formula

    print("Test 3, parse grounded formula")
    for k, v in beta_query.items():
        obj = parse_foq_formula(v)
        for _ in range(10):
            gv = random_p_ground(random_e_ground(v))
            obj.additive_ground(gv)
            print(f"[Test3][Meta Formula] {obj.meta_formula}")
            print(f"[Test3][Adder formula] {gv}")
            print(f"[Test3][Grounded Formula] {obj.ground_formula}")
