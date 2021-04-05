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
    def criterion(self, answer_emb: torch.Tensor, answer_estimation: List[IntList]):
        pass

class TransE_Tnorm(AppFOQEstimator, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_embeddings=100,
                                              embedding_dim=3)
        self.relation_embeddings = nn.Embedding(num_embeddings=100,
                                                embedding_dim=3)

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

    def criterion(self, answer_emb: torch.Tensor, answer_estimation: List[IntList]):
        return super().criterion(answer_emb, answer_estimation)