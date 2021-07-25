import logging
from collections import defaultdict

from fol.foq_v2 import binary_formula_iterator, parse_formula, projection_sink

if __name__ == "__main__":
    logging.basicConfig(filename='formula_generation.log',
                        filemode='wt',
                        level=logging.INFO)
    total_count = 0
    reductions = defaultdict(set)
    for k in range(1, 5):
        it = binary_formula_iterator(depth=4, num_anchor_nodes=k, parent=None)
        for i, f in enumerate(it):
            query = parse_formula(f)
            pf = query.formula
            reductions['pure_formula'].add(pf)
            psinkf = projection_sink(query).formula
            reductions['project_sink_formula'].add(psinkf)
            total_count += 1
            logging.info("record:"
                         "anchor_nodes\tidx\tformula\tpure_formula\tproject_sink:"
                         f"{k:5d}\t{i:5d}\t{f:50s}\t{pf:50s}\t{psinkf:50s}")
        for k, v in reductions.items():
            logging.info(f"statistics:{len(v)} {k} produced cumulatively")
    logging.info(f":statistics:{total_count} formulas are produced")