import os
import logging
from collections import defaultdict
import pandas as pd

from fol.foq_v2 import (concate_n_chains, copy_query,
                        negation_sink,
                        binary_formula_iterator,
                        concate_iu_chains,
                        parse_formula,
                        projection_sink, to_D,
                        union_bubble,
                        DeMorgan_replacement,
                        to_d,
                        transformation)


def convert_log_to_csv(logfile, outfile):
    already = False
    if os.path.exists(outfile):
        already = True
        already_df = pd.read_csv(outfile)
        formula_id_set = set(already_df.formula_id)
        original_set = set(already_df.original)
        outfile = outfile.replace(".csv", "_extend.csv")
    formula_id_set = set()
    original_set = ()

    data_dict = defaultdict(list)
    with open(logfile, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            *_, rtype, schema, data = line.split(":")
            row_data = dict()
            if rtype == 'record':
                for k, v in zip(schema.split('\t'), data.split('\t')):
                    row_data[k.strip()] = v.strip()
            
                if row_data['original'] in original_set:
                    continue
            
                if row_data['formula_id'] in formula_id_set:
                    num = int(row_data['formula_id'][-4:])
                    while True:
                        new_key = f"type{num+1:04d}"
                        if new_key not in formula_id_set:
                            row_data['formula_id'] = new_key
                            formula_id_set.add(new_key)
                            break
                        num += 1
                
                for k in row_data:
                    data_dict[k].append(row_data[k])

    df = pd.DataFrame(data_dict)
    df = df.drop_duplicates(subset=['original'])
    df.to_csv(outfile, index=False)
    if already:
        df = df.append(already_df, ignore_index=True)
        df.to_csv(outfile.replace("extend", "full"), index=False)
    for c in df.columns:
        logging.info(f"{len(df[c].unique())} {c} unique formulas found")
    # for i, row in df.iterrows():
        # if len(row) == len(set(row.tolist())):
            # print(row.tolist())


def convert_to_dnf(query):
    # query = transformation(query, projection_sink)
    def dnf_step(query):
        return union_bubble(concate_n_chains(negation_sink(query)))

    query = transformation(query, dnf_step)
    return query


def count_query_depth(query):
    if query.__o__ == 'e':
        return 0
    elif query.__o__ in 'uiUID':
        return max(count_query_depth(q) for q in query.sub_queries)
    elif query.__o__ == 'p':
        return count_query_depth(query.query) + 1
    elif query.__o__ == 'n':
        return count_query_depth(query.query)
    elif query.__o__ == 'd':
        return max(count_query_depth(query.lquery), count_query_depth(query.rquery))
    else:
        raise NotImplementedError


def normal_forms_generation(formula):
    result = {}
    query = parse_formula(formula)
    result['original_depth'] = count_query_depth(query)
    formula = query.formula
    # proj, rproj = load_graph()
    # query.backward_sample()
    result["original"] = query.formula
    query = DeMorgan_replacement(parse_formula(formula))
    DM_MultiI = concate_iu_chains(copy_query(query, True))
    result["DeMorgan"] = query.formula
    result["DeMorgan+MultiI"] = DM_MultiI.formula
    query_dnf = convert_to_dnf(parse_formula(formula))
    result["DNF"] = query_dnf.formula
    query = to_d(parse_formula(formula))
    result["diff"] = query.formula
    query = to_d(parse_formula(query_dnf.formula))
    result["DNF+diff"] = query.formula
    query_dnf_multiiu = concate_iu_chains(parse_formula(query_dnf.formula))
    result["DNF+MultiIU"] = query_dnf_multiiu.formula
    result["DNF+MultiIUd"] = concate_iu_chains(
                                parse_formula(result['DNF+diff']))
    query = to_D(parse_formula(query_dnf_multiiu.formula))
    result["DNF+MultiIUD"] = query.formula
    return result


if __name__ == "__main__":
    # formula = "(i,(i,(n,(p,(e))),(p,(i,(n,(p,(e))),(p,(e))))),(u,(p,(p,(e))),(p,(e))))"
    # r = normal_forms_generation(formula)
    # print(r)

    logging.basicConfig(filename='logs/formula_generation_test.log',
                        filemode='wt',
                        level=logging.INFO)
    total_count = 0
    reductions = defaultdict(set)

    for k in range(1, 6):
        it = binary_formula_iterator(depth=3, num_anchor_nodes=k)
        for i, f in enumerate(it):
            res = normal_forms_generation(f)
            res['formula_id'] = f"type{total_count:04d}"
            res['num_anchor_nodes'] = k
            keys = list(res.keys())
            title_str = "\t".join(keys)
            formula_str = "\t".join(str(res[k]) for k in keys)
            total_count += 1
            logging.info(f"record:{title_str}:{formula_str}")
            for _k in keys:
                reductions[_k].add(res[_k])

        convert_log_to_csv('logs/formula_generation_test.log',
                           f'logs/test_generated_formula_anchor_node={k}.csv')

        for k, v in reductions.items():
            logging.info(f"statistics:{len(v)} {k} produced cumulatively")
    logging.info(f":statistics:{total_count} formulas are produced")

