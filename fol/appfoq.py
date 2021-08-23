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


class TransEEstimator(AppFOQEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_embeddings=100,
                                              embedding_dim=3)
        self.relation_embeddings = nn.Embedding(num_embeddings=100,
                                                embedding_dim=3)
        self.loss_func = nn.MSELoss()

    def get_entity_embedding(self, entity_ids: torch.IntTensor):
        return self.entity_embeddings(entity_ids)

    def get_projection_embedding(self, proj_ids: torch.IntTensor, emb):
        assert emb.shape[0] == proj_ids.shape[0]
        return self.relation_embeddings(proj_ids) + emb

    def get_conjunction_embedding(self, conj_emb):
        pass

    def get_disjunction_embedding(self, disj_emb):
        pass

    def get_difference_embedding(self, lemb, remb):
        return torch.clamp(lemb - remb, 0)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]):
        batch_size = pred_emb.shape[0]
        loss = 0
        for b in range(batch_size):
            num_ans = len(answer_set[b])
            answer = torch.tensor(answer_set[b])
            a_emb = self.entity_embeddings(answer).reshape(-1, num_ans)
            p_emb = pred_emb[b].unsqueeze(-1)
            loss += self.loss_func(p_emb, a_emb)
        return loss


class CQDEstimator(AppFOQEstimator):
    def __init__(self, n_entity, n_relation, gamma, entity_dim, relation_dim, device,
                 norm_type, regulariser, init_size):
        super().__init__()
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = gamma
        self.epsilon = 2.0
        self.norm_type = norm_type
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.init_size = init_size
        self.entity_embeddings = nn.Embedding(num_embeddings=n_relation, embedding_dim=2 * self.entity_dim, sparse=True)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation, embedding_dim=2 * self.relation_dim,
                                                sparse=True)
        self.entity_embeddings.weight.data *= init_size
        self.relation_embeddings.weight.data *= init_size

    def get_entity_embedding(self, entity_ids: torch.Tensor):
        return self.entity_embeddings[entity_ids]

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        real_rel, imagine_rel = torch.chunk(self.relation_embeddings(proj_ids), 2, dim=-1)
        real_emb, imagine_emb = torch.chunk(emb, 2, dim=-1)
        real_query = real_emb * real_rel - imagine_emb * imagine_rel
        imagine_query = real_emb * imagine_rel + real_rel * imagine_emb
        return torch.cat([real_query, imagine_query], dim=-1)

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        if self.norm_type == 'min':
            emb = torch.min(torch.cat(conj_emb), dim=-1)
        elif self.norm_type == 'prod':
            emb = torch.prod(torch.cat(conj_emb), dim=-1)
        else:
            raise ValueError(f't_norm must be "min" or "prod", got {self.norm_type}')
        return emb

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        if self.norm_type == 'min':
            emb = torch.max(torch.cat(disj_emb), dim=-1)
        elif self.norm_type == 'prod':
            emb = torch.sum(torch.cat(disj_emb), dim=-1) - torch.prod()
        else:
            raise ValueError(f't_norm must be "min" or "prod", got {self.norm_type}')
        return emb

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union=False):
        real_allentity, imagine_allentity = torch.chunk(self.entity_embeddings.weight, 2, dim=-1)
        real_pred, imagine_pred = torch.chunk(pred_emb, 2, dim=-1)
        predictions = torch.matmul(real_pred, real_allentity.transpose(0, 1)) + \
                      torch.matmul(imagine_pred, imagine_allentity.transpose(0, 1))  # todo: note this is add
        entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        loss_predict = entropy_loss(predictions,
                                    self.get_entity_embedding(torch.tensor(answer_set, device=self.device)))
        '''
        regularisation_factors = self.sqrt()
        loss_regularisation = self.regulariser.forward()
        '''
        return loss_predict

