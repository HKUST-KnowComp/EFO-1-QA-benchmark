import logging

from fol.foq_v2 import binary_formula_iterator, parse_formula

if __name__ == "__main__":
    logging.basicConfig(filename='formula_generation.log',
                        filemode='wt',
                        level=logging.INFO)
    it = binary_formula_iterator(depth=4, num_anchor_nodes=4, root=True)
    for i, f in enumerate(it):
        logging.info(f"{i:10d}\t{f:20s}\t{parse_formula(f).formula:20s}")
