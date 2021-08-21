from collections import defaultdict
import os.path as osp
import os
from shutil import rmtree
from multiprocessing import Pool

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
    result['DeMorgan+MultiI'] = concate_iu_chains(copy_query(result["DeMorgan"], True))
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


def sample_by_row_final(row, easy_proj, hard_proj, hard_rproj):
    while True:
        query_instance = parse_formula(row.original)
        full_answers = query_instance.backward_sample(hard_proj, hard_rproj)
        easy_answers = query_instance.deterministic_query(easy_proj)
        hard_answers = full_answers.difference(easy_answers)
        results = normal_forms_transformation(query_instance)
        if 0 < len(hard_answers) <= 100:
            break
    return list(easy_answers), list(hard_answers), results


if __name__ == "__main__":
    df = pd.read_csv("logs/generated_formula_anchor_node=3.csv")
    beta_data_folders = ["data/FB15k-237-betae",
                         "data/FB15k-betae",
                         "data/NELL-betae"]
    for data_path in beta_data_folders:
        ent2id, rel2id, \
            proj_train, reverse_train, \
            proj_valid, reverse_valid, \
            proj_test, reverse_test = load_data_with_indexing(data_path)

        kg_name = osp.basename(data_path).replace("-betae", "")
        out_folder = osp.join("data", "benchmark", kg_name)
        if osp.exists(out_folder): rmtree(out_folder)
        os.makedirs(out_folder, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            fid = row.formula_id
            data = defaultdict(list)
            def sampler_func(i):
                row_data = {}
                easy_answers, hard_answers, results = sample_by_row_final(
                    row, proj_valid, proj_test, reverse_test)
                row_data['easy_answers'] = easy_answers
                row_data['hard_answers'] = hard_answers
                for k in results:
                    row_data[k] = results[k].dumps
                return row_data
 
            produced_size = 0
            sample_size = 5000
            generated = set()
            while produced_size < sample_size:
                with Pool(12) as p:
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
                         

            pd.DataFrame(data).to_csv(osp.join(out_folder, f"data-{fid}.csv"), index=False)
#           for i in tqdm(range(10000), leave=False, desc=row.original + fid):
#               query_id = f"{fid}-sample{i:04d}"
#               data['query_id'].append(query_id)
#               easy_answers, hard_answers, results = sample_by_row_final(
#                   row, proj_train, reverse_train, proj_test)
#               data['easy_answers'].append(easy_answers)
#               data['hard_answers'].append(hard_answers)
#               for k in results:
#                   data[k].append(results[k].dumps)

                
        

