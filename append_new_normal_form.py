import os
import pandas as pd
import json

from pandas.core.frame import DataFrame

from fol.foq_v2 import (concate_n_chains, copy_query,
                        negation_sink,
                        binary_formula_iterator,
                        concate_iu_chains,
                        decompose_D,
                        parse_formula,
                        projection_sink, to_D,
                        union_bubble,
                        DeMorgan_replacement,
                        to_d,
                        transformation)


def convert_query(df,
                  old_form_name='DNF+MultiIUD',
                  new_form_name='DNF+MultiIUd',
                  convert_functional=decompose_D):
    def convertor(f):
        query_instance = parse_formula(f)
        query_instance = convert_functional(query_instance)
        return query_instance.formula

    df[new_form_name] = df[old_form_name].map(convertor)
    return df

def convert_grounded_query(df,
                           old_form_name='DNF+MultiIUD',
                           new_form_name='DNF+MultiIUd',
                           old_form_formula=None,
                           convert_functional=None):
    assert old_form_formula is not None
    assert convert_functional is not None
    
    def grounded_convertor(f):
        query_instance = parse_formula(old_form_formula)
        query_instance.additive_ground(json.loads(f))
        query_instance = convert_functional(query_instance)
        return query_instance.dumps

    df[new_form_name] = df[old_form_name].map(grounded_convertor)
    return df

if __name__ == "__main__":
    target_folder = "data/benchmark-sample"
    formula_file = "logs/generated_formula_anchor_node=3.csv"
    df = pd.read_csv(formula_file)
    df = convert_query(df)
    df.to_csv(formula_file)
    

    for kg in ["FB15k-237", "FB15k", "NELL"]:
        folder = os.path.join(target_folder, kg)
        for i, row in df.iterrows():
            print(row.formula_id)
            data_file = f"data-{row.formula_id}.csv"
            data_df = pd.read_csv(os.path.join(folder, data_file))
            converted_data_df = convert_grounded_query(
                data_df,
                old_form_formula=row['DNF+MultiIUD'],
                convert_functional=decompose_D)
            converted_data_df.to_csv(os.path.join(folder, data_file))

