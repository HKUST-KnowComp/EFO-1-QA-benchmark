from collections import defaultdict
import os.path as osp
import os
from random import shuffle
from shutil import rmtree
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from fol.foq_v2 import (DeMorgan_replacement, concate_iu_chains, parse_formula, projection_sink,
                        to_d, to_D, copy_query, Entity, Projection, decompose_D)
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
    result["DNF+MultiIUd"] = decompose_D(copy_query(result["DNF+MultiIUD"], True))
    return result


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
        out_folder = osp.join("data", "benchmark_1p20k", kg_name)
        if osp.exists(out_folder): rmtree(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        formula = "(p,(e))"
        data = defaultdict(list)

        produced_size = 0
        sample_size = 22812
        generated = set()
        entity_keys = list(proj_test.keys())
        shuffle(entity_keys)
        for e in entity_keys:
            relation_keys = list(proj_test[e].keys())
            shuffle(relation_keys)
            for r in relation_keys:
                query = Entity()
                query.entities = [e]
                query = Projection(query)
                query.relations = [r]
                full_answers = query.deterministic_query(proj_test)
                easy_answers = query.deterministic_query(proj_valid)
                hard_answers = full_answers.difference(easy_answers)
                if len(hard_answers) > 100:
                    continue
                results = normal_forms_transformation(query)
                produced_size += 1
                print(produced_size)
                data['easy_answers'].append(list(easy_answers))
                data['hard_answers'].append(list(hard_answers))
                for k in results:
                    data[k].append(results[k].dumps)

                if produced_size == sample_size:
                    pd.DataFrame(data).to_csv(
                        osp.join(out_folder, "1p22812.csv"), index=False)
                    exit()
#           for i in tqdm(range(10000), leave=False, desc=row.original + fid):
#               query_id = f"{fid}-sample{i:04d}"
#               data['query_id'].append(query_id)
#               easy_answers, hard_answers, results = sample_by_row_final(
#                   row, proj_train, reverse_train, proj_test)
#               data['easy_answers'].append(easy_answers)
#               data['hard_answers'].append(hard_answers)
#               for k in results:
#                   data[k].append(results[k].dumps)

                
        

