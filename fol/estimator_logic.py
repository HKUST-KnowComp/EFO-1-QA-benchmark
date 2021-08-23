from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)


eps = 1e-6


def order_bounds(embedding):  # ensure lower < upper truth bound for logic embedding
    embedding = torch.clamp(embedding, 0, 1)
    lower, upper = torch.chunk(embedding, 2, dim=-1)
    contra = lower > upper
    if contra.any():  # contradiction
        mean = (lower + upper) / 2
        lower = torch.where(lower > upper, mean, lower)
        upper = torch.where(lower > upper, mean, upper)
    ordered_embedding = torch.cat([lower, upper], dim=-1)
    return ordered_embedding


def valclamp(x, a=1, b=6, lo=0, hi=1):  # relu1 with gradient-transparent clamp on negative
    elu_neg = a * (torch.exp(b * x) - 1)
    return ((x < lo).float() * (lo + elu_neg - elu_neg.detach()) +
            (lo <= x).float() * (x <= hi).float() * x +
            (hi < x).float())


class LogicIntersection(nn.Module):

    def __init__(self, dim, tnorm, bounded, use_att, use_gtrans):
        super(LogicIntersection, self).__init__()
        self.dim = dim
        self.tnorm = tnorm
        self.bounded = bounded
        self.use_att = use_att
        self.use_gtrans = use_gtrans  # gradient transparency

        if use_att:  # use attention with weighted t-norm
            self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)

            if bounded:
                self.layer2 = nn.Linear(2 * self.dim, self.dim)  # same weight for bound pair
            else:
                self.layer2 = nn.Linear(2 * self.dim, 2 * self.dim)

            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        if self.use_att:  # use attention with weighted t-norm
            layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, batch_size, 2 * dim)
            attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)
            attention = attention / torch.max(attention, dim=0, keepdim=True).values

            if self.bounded:  # same weight for bound pair
                attention = torch.cat([attention, attention], dim=-1)

            if self.tnorm == 'mins':  # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = attention * torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)

            elif self.tnorm == 'luk':  # Lukasiewicz t-norm
                embedding = 1 - torch.sum(attention * (1 - embeddings), dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6. / embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)

            elif self.tnorm == 'prod':  # product t-norm
                embedding = torch.prod(torch.pow(torch.clamp(embeddings, 0, 1) + eps, attention), dim=0)

        else:  # no attention
            if self.tnorm == 'mins':  # minimum / Godel t-norm
                smooth_param = -10  # smooth minimum
                min_weights = torch.exp(smooth_param * embeddings)
                embedding = torch.sum(min_weights * embeddings, dim=0) / torch.sum(min_weights, dim=0)
                if self.bounded:
                    embedding = order_bounds(embedding)

            elif self.tnorm == 'luk':  # Lukasiewicz t-norm
                embedding = 1 - torch.sum(1 - embeddings, dim=0)
                if self.use_gtrans:
                    embedding = valclamp(embedding, b=6. / embedding.shape[0])
                else:
                    embedding = torch.clamp(embedding, 0, 1)

            elif self.tnorm == 'prod':  # product t-norm
                embedding = torch.prod(embeddings, dim=0)

        return embedding


class LogicProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers, bounded):
        super(LogicProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bounded = bounded
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = torch.sigmoid(x)

        if self.bounded:
            lower, upper = torch.chunk(x, 2, dim=-1)
            upper = lower + upper * (1 - lower)
            x = torch.cat([lower, upper], dim=-1)

        return x


class SizePredict(nn.Module):
    def __init__(self, entity_dim):
        super(SizePredict, self).__init__()

        self.layer2 = nn.Linear(entity_dim, entity_dim // 4)
        self.layer1 = nn.Linear(entity_dim // 4, entity_dim // 16)
        self.layer0 = nn.Linear(entity_dim // 16, 1)

        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer0.weight)

    def forward(self, entropy_embedding):
        x = self.layer2(entropy_embedding)
        x = F.relu(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer0(x)
        x = torch.sigmoid(x)

        return x.squeeze()


class LogicEstimator(AppFOQEstimator):
    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, num_layers,
                 negative_sample_size, t_norm, bounded, use_att, use_gtrans, device):
        super().__init__()
        self.name = 'logic'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.t_norm, self.bounded = t_norm, bounded
        if self.bounded:
            lower = torch.rand((n_entity, self.entity_dim))
            upper = lower + torch.rand((n_entity, self.entity_dim)) * (1 - lower)
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.cat([lower, upper], dim=-1), freeze=False)
        else:
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.rand((n_entity, self.entity_dim * 2)),
                                                                  freeze=False)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        embedding_range = torch.tensor([(self.gamma + self.epsilon) / entity_dim]).to(self.device)
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())

        self.center_net = LogicIntersection(self.entity_dim, t_norm, bounded, use_att, use_gtrans)
        self.projection_net = LogicProjection(self.entity_dim * 2, self.relation_dim, hidden_dim, num_layers, bounded)

    def get_entity_embedding(self, entity_ids: torch.Tensor):
        emb = self.entity_embeddings(entity_ids)
        return emb

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = self.relation_embeddings(proj_ids)
        pro_emb = self.projection_net(emb, rel_emb)
        return pro_emb

    def get_negation_embedding(self, embedding: torch.Tensor):
        if self.bounded:
            lower_embedding, upper_embedding = torch.chunk(embedding, 2, dim=-1)
            embedding = torch.cat([1 - upper_embedding, 1 - lower_embedding], dim=-1)
        else:
            embedding = 1 - embedding
        return embedding

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        all_emb = torch.stack(conj_emb)
        emb = self.center_net(all_emb)
        return emb

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        return torch.stack(disj_emb, dim=1)
    '''
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        n_remb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding([lemb, n_remb])

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
        assert False, 'Do not use d in Logic'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        assert False, 'Do not use D in Logic'

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        pred_emb = pred_emb.unsqueeze(dim=-2)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        answer_embedding = self.get_entity_embedding(torch.tensor(chosen_ans, device=self.device))
        neg_embedding = self.get_entity_embedding(torch.tensor(chosen_false_ans, device=self.device).view(-1))  # n*dim
        neg_embedding = neg_embedding.view(-1, self.negative_size, 2 * self.entity_dim)  # batch*negative*dim
        if union:
            positive_union_logit = self.compute_logit(answer_embedding.unsqueeze(1), pred_emb)
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(neg_embedding.unsqueeze(1), pred_emb)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, pred_emb)
            negative_logit = self.compute_logit(neg_embedding, pred_emb)  # b*negative
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_logit(self, entity_embedding, query_embedding):
        if self.bounded:
            lower_embedding, upper_embedding = torch.chunk(entity_embedding, 2, dim=-1)
            query_lower_embedding, query_upper_embedding = torch.chunk(query_embedding, 2, dim=-1)

            lower_dist = torch.norm(lower_embedding - query_lower_embedding, p=1, dim=-1)
            upper_dist = torch.norm(query_upper_embedding - upper_embedding, p=1, dim=-1)

            logit = self.gamma - (lower_dist + upper_dist) / 2 / lower_embedding.shape[-1]
        else:
            logit = self.gamma - torch.norm(entity_embedding - query_embedding, p=1, dim=-1) / query_embedding.shape[-1]

        logit *= 100  # Todo: why *100

        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False) -> (torch.Tensor, torch.Tensor):
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        pred_emb = pred_emb.unsqueeze(-2)  # batch*(disj)*1*dim
        batch_num = find_optimal_batch(all_embedding, 
                                       query_dist=pred_emb, 
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0), pred_emb)
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), pred_emb)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit
