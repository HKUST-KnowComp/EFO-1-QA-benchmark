from collections import defaultdict
import os.path as osp
import os
from multiprocessing import Pool
import argparse
from numpy.core.records import _deprecate_shape_0_as_None

from tqdm import tqdm
import pandas as pd

from fol.foq_v2 import (DeMorgan_replacement, concate_iu_chains, parse_formula,
                        to_d, to_D, copy_query)
from formula_generation import convert_to_dnf
from utils.util import load_data_with_indexing


def normal_forms_transformation(query):
    result = {}
    # proj, rproj = load_graph()
    # query.backward_sample()
    result["original"] = query
    result["DeMorgan"] = DeMorgan_replacement(copy_query(result["original"], True))
    result["DNF"] = convert_to_dnf(copy_query(result["original"], True))
    result["diff"] = to_d(copy_query(result["original"], True))
    result["DNF+diff"] = to_d(copy_query(result["DNF"], True))
    result["DNF+MultiIU"] = concate_iu_chains(copy_query(result["DNF"], True))
    result["DNF+MultiIUD"] = to_D(copy_query(result["DNF+MultiIU"], True))
    return result


def sample_by_row(row, easy_proj, easy_rproj, hard_proj):
    query_instance = parse_formula(row.original)
    easy_answers = query_instance.backward_sample(easy_proj, easy_rproj)
    full_answers = query_instance.deterministic_query(hard_proj)
    hard_answers = full_answers.difference(easy_answers)
    results = normal_forms_transformation(query_instance)
    for k in results:
        assert results[k].formula == row[k]
        _full_answer = results[k].deterministic_query(hard_proj)
        assert _full_answer == full_answers
        _easy_answer = results[k].deterministic_query(easy_proj)
        assert _easy_answer == easy_answers
    return list(easy_answers), list(hard_answers), results


def sample_by_row_final(row, easy_proj, hard_proj, hard_rproj,):
    while True:
        query_instance = parse_formula(row.original)
        full_answers = query_instance.backward_sample(hard_proj, hard_rproj)
        easy_answers = query_instance.deterministic_query(easy_proj)
        hard_answers = full_answers.difference(easy_answers)
        results = normal_forms_transformation(query_instance)
        if 0 < len(hard_answers) <= 100:
            break
    return list(easy_answers), list(hard_answers), results


def get_query_dataframe(row, easy_proj, hard_proj, hard_rproj, cpus=12, sample_size=5000):
    def sampler_func(i):
        row_data = {}
        easy_answers, hard_answers, results = sample_by_row_final(
            row, easy_proj, hard_proj, hard_rproj)
        row_data['easy_answers'] = easy_answers
        row_data['hard_answers'] = hard_answers
        for k in results:
            row_data[k] = results[k].dumps
        return row_data

    fid = row.formula_id
    data = defaultdict(list)
    generated_original_dumps = set()
    produced_size = 0
    while produced_size < sample_size:
        with Pool(cpus) as p:
            gets = p.map(sampler_func,
                         list(range(sample_size - produced_size)))
            for row_data in gets:
                if row_data['original'] in generated_original_dumps:
                    continue
                else:
                    generated_original_dumps.add(
                        row_data['original']
                    )
                    produced_size += 1
                    for k in row_data:
                        data[k].append(row_data[k])

    pd.DataFrame(data).iloc[sample_size, :].to_csv(
        osp.join(out_folder, f"data-{fid}.csv"), index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--graph", type=str)


if __name__ == "__main__":
    df = pd.read_csv("logs/generated_formula_anchor_node=3.csv")
    beta_data_folders = {"fb15k-237": "data/FB15k-237-betae",
                         "fb15k": "data/FB15k-betae",
                         "nell": "data/NELL-betae"}

    args = parser.parse_args()
    data_path = beta_data_folders[args.graph]
    ent2id, rel2id, \
        proj_train, reverse_train, \
        proj_valid, reverse_valid, \
        proj_test, reverse_test = load_data_with_indexing(data_path)

    kg_name = osp.basename(data_path).replace("-betae", "")
    out_folder = osp.join("data", "benchmark", kg_name)
    os.makedirs(out_folder, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        get_query_dataframe(row, proj_valid, proj_test, reverse_test)
