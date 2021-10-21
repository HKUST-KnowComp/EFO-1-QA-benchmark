from collections import defaultdict
import os.path as osp
import argparse
import os
import json
from shutil import rmtree
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from fol.foq_v2 import (DeMorgan_replacement, concate_iu_chains, parse_formula,
                        to_d, to_D, decompose_D, copy_query)
from formula_generation import convert_to_dnf
from utils.util import load_data_with_indexing

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_name", type=str, default="benchmark")
parser.add_argument("--input_formula_file", type=str, default="outputs/test_generated_formula_anchor_node=3.csv")
parser.add_argument("--sample_size", default=5, type=int)
parser.add_argument("--knowledge_graph", action="append")
parser.add_argument("--ncpus", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--meaningful_difference_setting", type=str, default='mixed')


def normal_forms_transformation(query):
    result = {}
    # proj, rproj = load_graph()
    # query.backward_sample()
    result["original"] = query
    result["DeMorgan"] = DeMorgan_replacement(copy_query(result["original"], True))
    result['DeMorgan+MultiI'] = concate_iu_chains(copy_query(result["DeMorgan"], True))
    result["DNF"] = convert_to_dnf(copy_query(result["original"], True))
    result["diff"] = to_d(copy_query(result["original"], True))
    result["DNF+diff"] = to_d(copy_query(result["DNF"], True))
    result["DNF+MultiIU"] = concate_iu_chains(copy_query(result["DNF"], True))
    result['DNF+MultiIU'].sort_sub()
    result["DNF+MultiIUD"] = to_D(copy_query(result["DNF+MultiIU"], True))
    result["DNF+MultiIUd"] = decompose_D(copy_query(result["DNF+MultiIUD"], True))
    return result


def sample_by_row(row, easy_proj, easy_rproj, hard_proj, meaningful_difference: bool = False):
    query_instance = parse_formula(row.original)
    easy_answers = query_instance.backward_sample(easy_proj, easy_rproj, meaningful_difference=meaningful_difference)
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


def sample_by_row_final(row, easy_proj, hard_proj, hard_rproj, meaningful_difference_setting: str = 'mixed'):
    while True:
        query_instance = parse_formula(row.original)
        if meaningful_difference_setting == 'mixed':
            formula = query_instance.formula
            meaningful_difference = ('d' in formula or 'D' in formula or 'n' in formula)
        elif meaningful_difference_setting == 'fixed_True':
            meaningful_difference = True
        elif meaningful_difference_setting == 'fixed_False':
            meaningful_difference = False
        else:
            assert False, 'Invalid setting!'
        full_answers = query_instance.backward_sample(hard_proj, hard_rproj,
                                                      meaningful_difference=meaningful_difference)
        assert full_answers == query_instance.deterministic_query(hard_proj)
        easy_answers = query_instance.deterministic_query(easy_proj)
        hard_answers = full_answers.difference(easy_answers)
        results = normal_forms_transformation(query_instance)
        if 0 < len(hard_answers) <= 100:
            break
    # for key in results:
        # parse_formula(row[key]).additive_ground(json.loads(results[key].dumps))
    return list(easy_answers), list(hard_answers), results


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    df = pd.read_csv(args.input_formula_file)
    beta_data_folders = {"FB15k-237": "data/FB15k-237-betae",
                         "FB15k": "data/FB15k-betae",
                         "NELL": "data/NELL-betae"}
    for kg in args.knowledge_graph:
        data_path = beta_data_folders[kg]
        ent2id, rel2id, \
            proj_train, reverse_train, \
            proj_valid, reverse_valid, \
            proj_test, reverse_test = load_data_with_indexing(data_path)

        kg_name = osp.basename(data_path).replace("-betae", "")
        out_folder = osp.join("data", args.benchmark_name, kg_name)
        os.makedirs(out_folder, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            fid = row.formula_id
            data = defaultdict(list)

            if args.ncpus > 1:
                def sampler_func(i):
                    row_data = {}
                    easy_answers, hard_answers, results = sample_by_row_final(
                        row, proj_valid, proj_test, reverse_test,
                        meaningful_difference_setting=args.meaningful_difference_setting)
                    row_data['easy_answers'] = easy_answers
                    row_data['hard_answers'] = hard_answers
                    for k in results:
                        row_data[k] = results[k].dumps
                    return row_data

                produced_size = 0
                sample_size = args.num_samples
                generated = set()
                while produced_size < sample_size:
                    with Pool(args.ncpus) as p:
                        gets = p.map(sampler_func, list(range(sample_size - produced_size)))

                        for row_data in gets:
                            original = row_data['original']
                            if original in generated:
                                continue
                            else:
                                produced_size += 1
                                generated.add(original)

                            for k in row_data:
                                data[k].append(row_data[k])
            else:
                generated = set()
                sampled_query = 0
                while sampled_query < args.num_samples:
                    easy_answers, hard_answers, results = sample_by_row_final(
                        row, proj_valid, proj_test, reverse_test,
                        meaningful_difference_setting=args.meaningful_difference_setting)
                    if results['original'].dumps in generated:
                        continue
                    else:
                        generated.add(results['original'].dumps)
                        sampled_query += 1
                    data['easy_answers'].append(easy_answers)
                    data['hard_answers'].append(hard_answers)
                    for k in results:
                        data[k].append(results[k].dumps)

            pd.DataFrame(data).to_csv(osp.join(out_folder, f"data-{fid}.csv"), index=False)
