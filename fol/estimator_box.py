from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act),
                              dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


def identity(x):
    return x


class BoxEstimator(AppFOQEstimator):

    def __init__(self, n_entity, n_relation, gamma, entity_dim,
                 relation_dim, offset_activation, center_reg,
                 negative_sample_size, device):
        super().__init__()
        self.name = 'box'
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.negative_size = negative_sample_size
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.device = device
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / entity_dim]),
            requires_grad=False
        )
        self.entity_embeddings = nn.Embedding(
            num_embeddings=n_entity, embedding_dim=self.entity_dim)
        self.relation_embeddings = nn.Embedding(
            num_embeddings=n_relation, embedding_dim=self.relation_dim)
        self.offset_embeddings = nn.Embedding(
            num_embeddings=n_relation, embedding_dim=self.entity_dim)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -self.embedding_range.item(),
                         self.embedding_range.item())
        nn.init.uniform_(self.relation_embeddings.weight,
                         -self.embedding_range.item(),
                         self.embedding_range.item())
        nn.init.uniform_(self.offset_embeddings.weight,
                         0,
                         self.embedding_range.item())
        self.center_net = CenterIntersection(self.entity_dim)
        self.offset_net = BoxOffsetIntersection(self.entity_dim)
        self.cen_reg = center_reg
        if offset_activation == 'none':
            self.func = identity
        elif offset_activation == 'relu':
            self.func = F.relu
        elif offset_activation == 'softplus':
            self.func = F.softplus
        else:
            assert False, "No valid activation function!"

    def get_entity_embedding(self, entity_ids: torch.LongTensor):
        center_emb = self.entity_embeddings(entity_ids)
        offset_emb = torch.zeros_like(center_emb).to(self.device)
        return torch.cat((center_emb, offset_emb), dim=-1)

    def get_projection_embedding(self, proj_ids: torch.LongTensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = self.relation_embeddings(proj_ids)
        r_offset_emb = self.offset_embeddings(proj_ids)
        q_emb, q_off_emb = torch.chunk(emb, 2, dim=-1)
        q_emb = torch.add(q_emb, rel_emb)
        q_off_emb = torch.add(q_off_emb, self.func(r_offset_emb))
        return torch.cat((q_emb, q_off_emb), dim=-1)

    def get_negation_embedding(self, emb: torch.Tensor):
        assert False, "box cannot handle negation"

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        return torch.stack(disj_emb, dim=1)

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        assert False, "box cannot handle negation"

    def get_multiple_difference_embedding(self,
                                          emb: List[torch.Tensor],
                                          **kwargs):
        assert False, "box cannot handle negation"

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        sub_center_list, sub_offset_list = [], []
        for sub_emb in conj_emb:
            sub_center, sub_offset = torch.chunk(sub_emb, 2, dim=-1)
            sub_center_list.append(sub_center)
            sub_offset_list.append(sub_offset)
        new_center = self.center_net(torch.stack(sub_center_list))
        new_offset = self.offset_net(torch.stack(sub_offset_list))
        return torch.cat((new_center, new_offset), dim=-1)

    def criterion(self,
                  pred_emb: torch.Tensor,
                  answer_set: List[IntList],
                  union=False):
        pred_emb = pred_emb.unsqueeze(dim=-2)
        chosen_answer, chosen_false_answer, subsampling_weight = \
            inclusion_sampling(answer_set,
                               negative_size=self.negative_size,
                               entity_num=self.n_entity)
        positive_all_embedding = self.get_entity_embedding(
            torch.tensor(chosen_answer, device=self.device))  # b*d
        positive_embedding, _ = torch.chunk(
            positive_all_embedding, 2, dim=-1)
        neg_embedding = self.get_entity_embedding(
            torch.tensor(chosen_false_answer, device=self.device).view(-1))
        neg_embedding = neg_embedding.view(
            -1, self.negative_size, 2 * self.entity_dim)  # batch*n*dim
        negative_embedding, _ = torch.chunk(neg_embedding, 2, dim=-1)
        if union:
            positive_union_logit = self.compute_logit(
                positive_embedding.unsqueeze(1), pred_emb)
            positive_logit = torch.max(
                positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(
                negative_embedding.unsqueeze(1), pred_emb)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(positive_embedding, pred_emb)
            negative_logit = self.compute_logit(negative_embedding, pred_emb)
        return positive_logit, negative_logit, subsampling_weight.to(
                                                                self.device)

    def compute_logit(self, entity_emb, query_emb):
        query_center_embedding, query_offset_embedding = torch.chunk(
            query_emb, 2, dim=-1)
        delta = (entity_emb - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) \
                - self.cen_reg * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self,
                                 pred_emb: torch.Tensor,
                                 union=False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding, _ = torch.chunk(
            self.get_entity_embedding(all_entities), 2, dim=-1)
        pred_emb = pred_emb.unsqueeze(-2)
        batch_num = find_optimal_batch(all_embedding,
                                       query_dist=pred_emb,
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(
                    answer_part.unsqueeze(0).unsqueeze(0), pred_emb)
                    # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0),
                                                pred_emb)
                                                # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit
