from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .appfoq import (AppFOQEstimator, IntList, inclusion_sampling,
                     find_optimal_batch)



class Regularizer:
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class BetaProjection(nn.Module):
    def __init__(self,
                 entity_dim,
                 relation_dim,
                 hidden_dim,
                 projection_regularizer,
                 num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(
            self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(
            self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(
                self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class BetaIntersection(nn.Module):
    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        # (num_conj, batch_size, 2 * dim)
        layer1_act = F.relu(self.layer1(all_embeddings))
        # (num_conj, batch_size, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)
        return alpha_embedding, beta_embedding


class BetaEstimator(AppFOQEstimator):
    name = "beta"

    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, num_layers,
                 negative_sample_size, device):
        super().__init__()
        self.name = 'beta'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.entity_embeddings = nn.Embedding(
            num_embeddings=n_entity,
            embedding_dim=self.entity_dim * 2)
        self.relation_embeddings = nn.Embedding(
            num_embeddings=n_relation,
            embedding_dim=self.relation_dim)
        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / entity_dim]),
            requires_grad=False
        ).to(self.device)
        nn.init.uniform_(tensor=self.entity_embeddings.weight,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embeddings.weight,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())

        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        # self.intersection_net = BetaIntersection(self.entity_dim)
        self.center_net = BetaIntersection(self.entity_dim)
        self.projection_net = BetaProjection(self.entity_dim * 2,
                                             self.relation_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)

    def get_entity_embedding(self, entity_ids: torch.LongTensor):
        emb = self.entity_embeddings(entity_ids)
        return self.entity_regularizer(emb)

    def get_projection_embedding(self, proj_ids: torch.LongTensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = self.relation_embeddings(proj_ids)
        pro_emb = self.projection_net(emb, rel_emb)
        return pro_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        sub_alpha_list, sub_beta_list = [], []
        for sub_emb in conj_emb:
            sub_alpha, sub_beta = torch.chunk(sub_emb, 2, dim=-1)
            sub_alpha_list.append(sub_alpha)  # b*dim
            sub_beta_list.append(sub_beta)
        all_alpha = torch.stack(sub_alpha_list)  # conj*b*dim
        all_beta = torch.stack(sub_beta_list)
        new_alpha, new_beta = self.center_net(all_alpha, all_beta)
        embedding = torch.cat([new_alpha, new_beta], dim=-1)
        return embedding

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        union_emb = torch.stack(disj_emb, dim=1)  # datch*disj*d
        return union_emb

    def get_negation_embedding(self, emb: torch.Tensor):
        return 1. / emb

    # a-b = a and(-b)
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        r_neg_emb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding([lemb, r_neg_emb])

    def criterion(self,
                  pred_emb: torch.Tensor,
                  answer_set: List[IntList],
                  union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        pred_emb = pred_emb.unsqueeze(dim=-2)   # batch*(disj)*1*dim
        alpha_embedding, beta_embedding = torch.chunk(pred_emb, 2, dim=-1)
        query_dist = torch.distributions.beta.Beta(
            alpha_embedding, beta_embedding)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size,
                               entity_num=self.n_entity)  # todo: negative
        answer_embedding = self.get_entity_embedding(
            torch.tensor(chosen_ans, device=self.device))  # batch*1*dim
        all_neg_emb = self.get_entity_embedding(torch.tensor(
            chosen_false_ans, device=self.device).view(-1))
        # batch*negative*dim
        all_neg_emb = all_neg_emb.view(-1,
                                       self.negative_size, 2 * self.entity_dim)
        if union:
            positive_union_logit = self.compute_logit(
                answer_embedding.unsqueeze(1), query_dist)  # b*disj
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(
                all_neg_emb.unsqueeze(1), query_dist)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, query_dist)
            negative_logit = self.compute_logit(
                all_neg_emb, query_dist)  # b*negative
        return positive_logit, negative_logit, subsampling_weight.to(
                                                        self.device)

    def compute_logit(self, entity_emb, query_dist):
        entity_alpha, entity_beta = torch.chunk(entity_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        logit = self.gamma - \
            torch.norm(torch.distributions.kl.kl_divergence(
                entity_dist, query_dist), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self,
                                 pred_emb: torch.Tensor,
                                 union: bool = False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        pred_alpha, pred_beta = torch.chunk(
            pred_emb, 2, dim=-1)  # batch*(disj)*dim
        query_dist = torch.distributions.beta.Beta(
            pred_alpha.unsqueeze(-2), pred_beta.unsqueeze(-2))
        batch_num = find_optimal_batch(all_embedding,
                                       query_dist=query_dist,
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(
                    answer_part.unsqueeze(0).unsqueeze(0),
                    query_dist)  # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(
                    answer_part.unsqueeze(dim=0),
                    query_dist)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit


class BetaEstimator4V(AppFOQEstimator):
    """Beta embedding for verification
    """
    name = "beta"

    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, num_layers,
                 negative_sample_size, device):
        super().__init__()
        self.name = 'beta'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        # self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
        #   embedding_dim=self.entity_dim * 2)
        # self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
        # embedding_dim=self.relation_dim)
        self.entity_embedding = nn.Parameter(
            torch.zeros(n_entity, self.entity_dim * 2))
        self.relation_embedding = nn.Parameter(
            torch.zeros(n_relation, self.relation_dim))
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / entity_dim]),
            requires_grad=False
        )
        nn.init.uniform_(tensor=self.entity_embedding,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embedding,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())

        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        # self.intersection_net = BetaIntersection(self.entity_dim)
        self.center_net = BetaIntersection(self.entity_dim)
        self.projection_net = BetaProjection(self.entity_dim * 2,
                                             self.relation_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)

    def get_entity_embedding(self, entity_ids: torch.LongTensor,
                             **kwargs):
        # emb = self.entity_embedding[entity_ids, :]
        emb = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=entity_ids.view(-1)
        ).view(list(entity_ids.shape) + [self.entity_dim * 2])
        return self.entity_regularizer(emb)

    def get_projection_embedding(self, proj_ids: torch.LongTensor, emb,
                                 **kwargs):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=proj_ids.view(-1)).view(
                list(proj_ids.shape) + [self.relation_dim])
        pro_emb = self.projection_net(emb, rel_emb)
        return pro_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor],
                                  **kwargs):
        sub_alpha_list, sub_beta_list = [], []
        for sub_emb in conj_emb:
            sub_alpha, sub_beta = torch.chunk(sub_emb, 2, dim=-1)
            sub_alpha_list.append(sub_alpha)  # b*dim
            sub_beta_list.append(sub_beta)
        all_alpha = torch.stack(sub_alpha_list)  # conj*b*dim
        all_beta = torch.stack(sub_beta_list)
        new_alpha, new_beta = self.center_net(all_alpha, all_beta)
        embedding = torch.cat([new_alpha, new_beta], dim=-1)
        return embedding

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor],
                                  **kwargs):
        union_emb = torch.stack(disj_emb, dim=1)  # datch*disj*d
        return union_emb

    def get_negation_embedding(self, emb: torch.Tensor, **kwargs):
        return 1. / emb

    '''
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor,
                                 **kwargs):  # a-b = a and(-b)
        r_neg_emb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding([lemb, r_neg_emb])

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        lemb, remb_list = emb[0], emb[1:]
        emb_list = [lemb]
        for remb in remb_list:
            neg_remb = self.get_negation_embedding(remb, **kwargs)
            emb_list.append(neg_remb)
        return self.get_conjunction_embedding(emb_list, **kwargs)
        '''

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor,
                                 **kwargs):
        assert False, 'Do not use d in BetaE'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        assert False, 'Do not use D in BetaE'

    def criterion(self,
                  pred_emb: torch.Tensor,
                  answer_set: List[IntList],
                  union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        alpha_embedding, beta_embedding = torch.chunk(pred_emb, 2, dim=-1)
        query_dist = torch.distributions.beta.Beta(
            alpha_embedding, beta_embedding)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size,
                               entity_num=self.n_entity)  # todo: negative
        answer_embedding = self.get_entity_embedding(
            torch.tensor(chosen_ans, device=self.device)).squeeze()
        if union:
            positive_union_logit = self.compute_logit(
                answer_embedding.unsqueeze(1), query_dist)  # b*disj
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, query_dist)
        all_neg_emb = self.get_entity_embedding(torch.tensor(
            chosen_false_ans, device=self.device).view(-1))
        # batch*negative*dim
        all_neg_emb = all_neg_emb.view(-1,
                                       self.negative_size, 2 * self.entity_dim)
        if union:
            query_dist_unsqueezed = torch.distributions.beta.Beta(
                alpha_embedding.unsqueeze(2), beta_embedding.unsqueeze(2))
            negative_union_logit = self.compute_logit(
                all_neg_emb.unsqueeze(1), query_dist_unsqueezed)
            negative_logit = torch.max(negative_union_logit, dim=1)
        else:
            query_dist_unsqueezed = torch.distributions.beta.Beta(
                alpha_embedding.unsqueeze(1), beta_embedding.unsqueeze(1))
            negative_logit = self.compute_logit(
                all_neg_emb, query_dist_unsqueezed)  # b*negative
        return positive_logit, negative_logit, subsampling_weight.to(
                                                                self.device)

    def compute_logit(self, entity_emb, query_dist):
        entity_alpha, entity_beta = torch.chunk(entity_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        logit = self.gamma - \
            torch.norm(torch.distributions.kl.kl_divergence(
                entity_dist, query_dist), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self,
                                 pred_emb: torch.Tensor,
                                 union: bool = False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        pred_alpha, pred_beta = torch.chunk(
            pred_emb, 2, dim=-1)  # batch*(disj)*dim
        query_dist = torch.distributions.beta.Beta(
            pred_alpha.unsqueeze(-2), pred_beta.unsqueeze(-2))
        batch_num = find_optimal_batch(all_embedding,
                                       query_dist=query_dist,
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(
                    answer_part.unsqueeze(0).unsqueeze(0),
                    query_dist)  # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(
                    dim=0), query_dist)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit
