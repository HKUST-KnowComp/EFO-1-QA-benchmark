from abc import ABC, abstractclassmethod
from typing import List

import torch
from torch import nn


IntList = List[int]


class AppFOQEstimator(ABC):

    @abstractclassmethod
    def get_entity_embedding(self, entity_ids: IntList):
        pass

    @abstractclassmethod
    def get_projection_embedding(self, proj_ids: IntList, emb):
        pass

    @abstractclassmethod
    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractclassmethod
    def get_disjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractclassmethod
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractclassmethod
    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList])-> torch.Tensor:
        pass

class TransE_Tnorm(AppFOQEstimator, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_embeddings=100,
                                              embedding_dim=3)
        self.relation_embeddings = nn.Embedding(num_embeddings=100,
                                                embedding_dim=3)
        self.loss_func = nn.MSELoss()

    def get_entity_embedding(self, entity_ids: IntList):
        xe = torch.tensor(entity_ids)
        return self.entity_embeddings(xe)

    def get_projection_embedding(self, proj_ids: IntList, emb):
        assert emb.shape[0] == len(proj_ids)
        xp = torch.tensor(proj_ids)
        return self.relation_embeddings(xp) + emb

    def get_conjunction_embedding(self, lemb, remb):
        assert lemb.shape
        return torch.minimum(lemb, remb)

    def get_disjunction_embedding(self, lemb, remb):
        return torch.maximum(lemb, remb)

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