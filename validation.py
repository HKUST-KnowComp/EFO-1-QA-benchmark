from typing import Dict, Set
import logging
import argparse
from torch.utils.data.dataloader import DataLoader
import yaml

from KGReasoning.models import KGReasoning
from KGReasoning.dataloader import (SingledirectionalOneShotIterator,
                                    TrainDataset)
                                    
from fol.appfoq import BetaEstimator


parser = argparse.ArgumentParser()

parser.add_argument("--save_intermediate_results",
                    action="store_true",
                    type="bool",
                    default=False)
parser.add_argument("--begin_checkpoint",
                    type=str,
                    default="")
parser.add_argument("--model_config_file",
                    type=str,
                    default="config/validation.yaml")
parser.add_argument("--cuda", type=bool, default=False)


def ref_train_step(beta_train_queries: Dict[str, Set],
                   beta_train_answers: Dict[str, Set],
                   model: KGReasoning, 
                   optimizer,
                   nentity,
                   nrelation,
                   batch_size,
                   negative_sample_size) -> Dict:
    """
    Conduct a train step from KG reasoning
    """
    train_iterator = SingledirectionalOneShotIterator(DataLoader(
        TrainDataset(beta_train_queries,
                     nentity,
                     nrelation,
                     negative_sample_size,
                     beta_train_answers),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=TrainDataset.collate_fn
    ))
    log = model.train_step(model, optimizer, train_iterator, args)
    return log


def our_train_step(batch_data, formula, model: BetaEstimator, optimizer) -> Dict:
    pass


# def ref_eval_step(batch_data, model) -> Dict:
    # pass


# def out_eval_step(batch_data, model) -> Dict:
    # pass


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.model_config_file, 'rt') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    ref_model_config = model_config['ref']
