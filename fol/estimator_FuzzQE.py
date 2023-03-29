from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)


class Regular_sig(nn.Module):
    def __init__(self):
        super(Regular_sig, self).__init__()
        self.w = nn.Parameter(torch.tensor([8.0]))
        self.b = nn.Parameter(torch.tensor([-4.0]))

    def __call__(self, input):
        output = torch.sigmoid(input * self.w + self.b)
        return output


class bounded_01:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding, self.min_val, self.max_val)


class projection_net(nn.Module):

    def __init__(self, entity_dim, n_relation, rel_num_base, device):
        super(projection_net, self).__init__()
        self.n_relation = n_relation
        self.n_base = rel_num_base
        self.hidden_dim = entity_dim
        self.device = device

        self.ln0 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        self.rel_att = nn.Parameter(
            torch.zeros([n_relation, self.n_base], device=self.device))
        self.rel_base = nn.Parameter(
            torch.zeros(
                [self.n_base, self.hidden_dim, self.hidden_dim],
                device=self.device)
        )
        self.rel_bias = nn.Parameter(
            torch.zeros([self.n_base, self.hidden_dim], device=self.device))
        nn.init.orthogonal_(self.rel_base)
        nn.init.xavier_normal_(self.rel_att)
        nn.init.xavier_normal_(self.rel_bias)

    def forward(self, emb, proj_ids):
        project_r = torch.einsum('br,rio->bio', self.rel_att[proj_ids], self.rel_base)
        bias = torch.einsum('br,ri->bi', self.rel_att[proj_ids], self.rel_bias)
        output = torch.einsum('bio,bi->bo', project_r, emb) + bias
        output = self.ln0(output)
        return output  # activate is out!


class FuzzQEstiamtor(AppFOQEstimator):
    name = "FuzzQE"

    def __init__(self, n_entity, n_relation, gamma, negative_sample_size,
                 entity_dim, relation_num_base, t_norm, regular, device, projection_num=1,
                 conjunction_num=1):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.entity_dim = entity_dim
        self.rel_num_base = relation_num_base
        # equal model's gamma_off
        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.device = device
        t_norm_list = ["Lukasiewicz", "Godel", "product"]
        if t_norm not in t_norm_list:
            assert "does't support this t_norm"
        self.t_norm = t_norm
        if regular == "sigmoid":
            self.p_regular = Regular_sig()
            self.e_regular = Regular_sig()
        elif regular == "bounded":
            self.p_regular = bounded_01(0.0, 1.0)
            self.e_regular = bounded_01(0.0, 1.0)
        else:
            assert "does't support this activate function"

        self.entity_embedding = nn.Parameter(
            torch.zeros(n_entity, self.entity_dim, device=self.device))
        nn.init.uniform_(tensor=self.entity_embedding, a=0.0, b=1.0)
        self.projection_net = projection_net(
            entity_dim, n_relation, relation_num_base, self.device)

    def get_entity_embedding(self, entity_ids: torch.Tensor, freeze=False, meta_parameters=None):
        emb = torch.index_select(self.entity_embedding,
                                 dim=0,
                                 index=entity_ids.view(-1)
                                 )
        return self.e_regular(emb)

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb, net_symbol=-1, freeze_rel=False, freeze_proj=False,
                                 meta_parameters=None):
        proj_emb = self.p_regular(self.projection_net(emb, proj_ids))
        return proj_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor], net_symbol=-1, freeze=False,
                                  meta_parameters=None):
        """
        The layernorm is ignored since we only use Godel
        """
        embeddings = torch.stack(conj_emb)
        if self.t_norm == "Lukasiewicz":
            x = torch.sum(embeddings, dim=0) - embeddings.shape[0] + 1
            x = F.relu(x)
            return x
        elif self.t_norm == "Godel":
            x, _ = torch.min(embeddings, dim=0)
            return x
        elif self.t_norm == "product":
            x = torch.prod(embeddings, dim=0)
            return x
        else:
            assert "does't support this t_norm"

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor], freeze=False, meta_parameters=None):
        union_emb, _ = torch.max(torch.stack(disj_emb), dim=0)  # batch*dim

        return union_emb

    def get_negation_embedding(self, emb: torch.Tensor, freeze=False, meta_parameters=None):

        return 1. - emb

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor, meta_parameters=None):
        assert False, 'Do not use d in Fuzzle'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], kwargs):
        assert False, 'Do not use D in Fuzzle'

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], freeze_entity=False, union: bool = False,
                  meta_parameters=None):
        assert pred_emb.shape[0] == len(answer_set)
        query_emb = pred_emb.unsqueeze(1)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size,
                               entity_num=self.n_entity)  # todo: negative
        answer_embedding = self.get_entity_embedding(torch.tensor(np.array(chosen_ans), device=self.device), ).unsqueeze(1)

        positive_logit = self.compute_logit(answer_embedding, query_emb)
        all_neg_emb = self.get_entity_embedding(torch.tensor(
            np.array(chosen_false_ans), device=self.device).view(-1), )
        # batch*negative*dim
        all_neg_emb = all_neg_emb.view(-1,
                                       self.negative_size, self.entity_dim)
        negative_logit = self.compute_logit(all_neg_emb, query_emb)
        return positive_logit, negative_logit, subsampling_weight.to(
            self.device)

    def compute_logit(self, entity_emb, query_emb):
        cos = nn.CosineSimilarity(dim=-1)
        logit = self.gamma * cos(entity_emb, query_emb)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False, meta_parameters=None):
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities, )  # nentity*dim
        batch_num = find_optimal_batch(all_embedding,
                                       query_dist=pred_emb.unsqueeze(1),
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            logit_part = self.compute_logit(answer_part.unsqueeze(0), pred_emb.unsqueeze(1))  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit

