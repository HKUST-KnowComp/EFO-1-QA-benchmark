import logging
from collections import defaultdict
import pandas as pd
from pandas.core.reshape.concat import concat

from fol.foq_v2 import (DeMorgan_rule, binary_formula_iterator, concate_iu_chains, parse_formula,
                        projection_sink, union_bubble)


def convert_log_to_csv(logfile):
    data_dict = defaultdict(list)
    with open(logfile, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            *_, rtype, schema, data = line.split(":")
            if rtype == 'record':
                for k, v in zip(schema.split('\t'), data.split('\t')):
                    data_dict[k.strip()].append(v.strip())
    pd.DataFrame(data_dict).to_csv(logfile.replace('.log', '.csv'),
                                   index=False)


if __name__ == "__main__":
    logging.basicConfig(filename='formula_generation.log',
                        filemode='wt',
                        level=logging.INFO)
    total_count = 0
    reductions = defaultdict(set)
    for k in range(1, 5):
        it = binary_formula_iterator(depth=3, num_anchor_nodes=k)
        for i, f in enumerate(it):
            query = parse_formula(f)
            pf = query.formula
            reductions['pure_formula'].add(pf)

            query = projection_sink(query)
            psinkf = query.formula
            reductions['project_sink_formula'].add(psinkf)

            query = DeMorgan_rule(query)
            demorganf = query.formula
            reductions['DeMorgan_formula'].add(demorganf)

            query = union_bubble(query)
            dnf = query.formula
            reductions['dnf'].add(dnf)

            query = concate_iu_chains(query)
            compact_dnf = query.formula
            reductions['compact_dnf'].add(dnf)

            total_count += 1
            logging.info("record:"
                         "anchor_nodes\tidx\tformula\tpure\tproject_sink\tDeMorgan\tDNF\tCompactDNF:"
                         f"{k:5d}\t{i:5d}\t{f:50s}\t{pf:50s}\t{psinkf:50s}\t{demorganf:50s}\t{dnf}\t{compact_dnf}")
        for k, v in reductions.items():
            logging.info(f"statistics:{len(v)} {k} produced cumulatively")
    logging.info(f":statistics:{total_count} formulas are produced")

    convert_log_to_csv('formula_generation.log')
