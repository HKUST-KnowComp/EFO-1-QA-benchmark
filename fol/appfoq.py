from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

IntList = List[int]
eps = 1e-6


def find_optimal_batch(answer_emb: torch.tensor, query_dist: torch.tensor, compute_logit, union: bool = False):
    batch_num = 1
    while True:
        try:
            batch_size = int(answer_emb.shape[0] / batch_num)
            batch_answer_emb = answer_emb[0:batch_size]
            if union:
                logit = compute_logit(batch_answer_emb.unsqueeze(0).unsqueeze(0), query_dist)
            else:
                logit = compute_logit(batch_answer_emb.unsqueeze(0), query_dist)
            return batch_num * 2
        except RuntimeError:
            batch_num *= 2


def negative_sampling(answer_set: List[IntList], negative_size: int, entity_num: int, k=1, base_num=4):
    all_chosen_ans = []
    all_chosen_false_ans = []
    subsampling_weight = torch.zeros(len(answer_set))
    for i in range(len(answer_set)):
        all_chosen_ans.append(random.choices(answer_set[i], k=k))
        subsampling_weight[i] = len(answer_set[i]) + base_num
        now_false_ans_size = 0
        negative_sample_list = []
        while now_false_ans_size < negative_size:
            negative_sample = np.random.randint(
                entity_num, size=negative_size * 2)
            mask = np.in1d(
                negative_sample,
                answer_set[i],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            now_false_ans_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[
                          :negative_size]
        all_chosen_false_ans.append(negative_sample)
    subsampling_weight = torch.sqrt(1 / subsampling_weight)
    return all_chosen_ans, all_chosen_false_ans, subsampling_weight


def inclusion_sampling(answer_set: List[IntList], negative_size: int, entity_num: int, k=1, base_num=4):
    all_chosen_ans = []
    all_chosen_false_ans = []
    subsampling_weight = torch.zeros(len(answer_set))
    for i in range(len(answer_set)):
        all_chosen_ans.append(random.choices(answer_set[i], k=k))
        subsampling_weight[i] = len(answer_set[i]) + base_num
        negative_sample = np.random.randint(entity_num, size=negative_size)
        all_chosen_false_ans.append(negative_sample)
    subsampling_weight = torch.sqrt(1 / subsampling_weight)
    return all_chosen_ans, all_chosen_false_ans, subsampling_weight


def compute_final_loss(positive_logit, negative_logit, subsampling_weight):
    positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)   # note this is b*1 by beta
    negative_score = F.logsigmoid(-negative_logit)
    negative_score = torch.mean(negative_score, dim=1)
    positive_loss = -(positive_score * subsampling_weight).sum()
    negative_loss = -(negative_score * subsampling_weight).sum()
    positive_loss /= subsampling_weight.sum()
    negative_loss /= subsampling_weight.sum()
    return positive_loss, negative_loss


def compute_loss_bpr(positive_logit, negative_logit, subsampling_weight):
    diff = -F.logsigmoid(positive_logit - negative_logit)
    unweighted_sample_loss = torch.mean(diff, dim=-1)
    loss = (subsampling_weight * unweighted_sample_loss).sum()
    loss /= subsampling_weight.sum()
    return loss


class AppFOQEstimator(ABC, nn.Module):

    @abstractmethod
    def get_entity_embedding(self, entity_ids: torch.Tensor):
        pass

    @abstractmethod
    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        pass

    @abstractmethod
    def get_negation_embedding(self, emb: torch.Tensor):
        pass

    @abstractmethod
    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        pass

    @abstractmethod
    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        pass

    @abstractmethod
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractmethod
    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union: bool = False):
        pass

    @abstractmethod
    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False):
        pass


