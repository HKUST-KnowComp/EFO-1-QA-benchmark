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


class Regularizer:
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
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
        self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
                                              embedding_dim=self.entity_dim * 2)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        embedding_range = torch.tensor([(self.gamma + self.epsilon) / entity_dim]).to(self.device)
        nn.init.uniform_(tensor=self.entity_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        self.intersection_net = BetaIntersection(self.entity_dim)
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
        new_alpha, new_beta = self.intersection_net(all_alpha, all_beta)
        embedding = torch.cat([new_alpha, new_beta], dim=-1)
        return embedding

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        union_emb = torch.stack(disj_emb, dim=1)  # datch*disj*d
        return union_emb

    def get_negation_embedding(self, emb: torch.Tensor):
        return 1. / emb

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):  # a-b = a and(-b)
        r_neg_emb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding([lemb, r_neg_emb])

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        pred_emb = pred_emb.unsqueeze(dim=-2)   # batch*(disj)*1*dim
        alpha_embedding, beta_embedding = torch.chunk(pred_emb, 2, dim=-1)
        query_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)  # todo: negative
        answer_embedding = self.get_entity_embedding(torch.tensor(chosen_ans, device=self.device))  # batch*1*dim
        all_neg_emb = self.get_entity_embedding(torch.tensor(chosen_false_ans, device=self.device).view(-1))
        all_neg_emb = all_neg_emb.view(-1, self.negative_size, 2 * self.entity_dim)  # batch*negative*dim
        if union:
            positive_union_logit = self.compute_logit(answer_embedding.unsqueeze(1), query_dist)  # b*disj
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(all_neg_emb.unsqueeze(1), query_dist)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, query_dist)
            negative_logit = self.compute_logit(all_neg_emb, query_dist)  # b*negative
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_logit(self, entity_emb, query_dist):
        entity_alpha, entity_beta = torch.chunk(entity_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        pred_alpha, pred_beta = torch.chunk(pred_emb, 2, dim=-1)  # batch*(disj)*dim
        query_dist = torch.distributions.beta.Beta(pred_alpha.unsqueeze(-2), pred_beta.unsqueeze(-2))
        batch_num = find_optimal_batch(all_embedding, query_dist=query_dist,
                                       compute_logit=self.compute_logit, union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0),
                                                query_dist)  # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), query_dist)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit


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
                 relation_dim, offset_activation, center_reg, negative_sample_size, device):
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
        self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
                                              embedding_dim=self.entity_dim)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        self.offset_embeddings = nn.Embedding(num_embeddings=n_relation, embedding_dim=self.entity_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -self.embedding_range.item(), self.embedding_range.item())
        nn.init.uniform_(self.relation_embeddings.weight, -self.embedding_range.item(), self.embedding_range.item())
        nn.init.uniform_(self.offset_embeddings.weight, 0, self.embedding_range.item())
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
        rel_emb, r_offset_emb = self.relation_embeddings(proj_ids), self.offset_embeddings(proj_ids)
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

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        sub_center_list, sub_offset_list = [], []
        for sub_emb in conj_emb:
            sub_center, sub_offset = torch.chunk(sub_emb, 2, dim=-1)
            sub_center_list.append(sub_center)
            sub_offset_list.append(sub_offset)
        new_center = self.center_net(torch.stack(sub_center_list))
        new_offset = self.offset_net(torch.stack(sub_offset_list))
        return torch.cat((new_center, new_offset), dim=-1)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union=False):
        pred_emb = pred_emb.unsqueeze(dim=-2)
        chosen_answer, chosen_false_answer, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        positive_all_embedding = self.get_entity_embedding(torch.tensor(chosen_answer, device=self.device))  # b*d
        positive_embedding, _ = torch.chunk(positive_all_embedding, 2, dim=-1)
        neg_embedding = self.get_entity_embedding(torch.tensor(chosen_false_answer, device=self.device).view(-1))
        neg_embedding = neg_embedding.view(-1, self.negative_size, 2 * self.entity_dim)  # batch*n*dim
        negative_embedding, _ = torch.chunk(neg_embedding, 2, dim=-1)
        if union:
            positive_union_logit = self.compute_logit(positive_embedding.unsqueeze(1), pred_emb)
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(negative_embedding.unsqueeze(1), pred_emb)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(positive_embedding, pred_emb)
            negative_logit = self.compute_logit(negative_embedding, pred_emb)
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_logit(self, entity_emb, query_emb):
        query_center_embedding, query_offset_embedding = torch.chunk(query_emb, 2, dim=-1)
        delta = (entity_emb - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen_reg * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union=False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding, _ = torch.chunk(self.get_entity_embedding(all_entities), 2, dim=-1)
        pred_emb = pred_emb.unsqueeze(-2)
        batch_num = find_optimal_batch(all_embedding, query_dist=pred_emb,
                                       compute_logit=self.compute_logit, union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0),
                                                pred_emb)  # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), pred_emb)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit


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

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        n_remb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding([lemb, n_remb])

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
        batch_num = find_optimal_batch(all_embedding, query_dist=pred_emb, compute_logit=self.compute_logit,
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


class TwoLayerNet(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        return self.layer2(F.relu(self.layer1(emb)))


class NLKProjection(nn.Module):
    def __init__(self, dim, hidden_dim, group_num):
        super(NLKProjection, self).__init__()
        self.dim, self.hidden_dim, self.concat_dim = dim, hidden_dim, 2 * dim + group_num
        self.MLP1 = TwoLayerNet(dim, hidden_dim, dim)
        self.MLP2 = TwoLayerNet(dim, hidden_dim, dim)
        self.MLP3 = TwoLayerNet(self.concat_dim, hidden_dim, dim)
        self.MLP4 = TwoLayerNet(self.concat_dim, hidden_dim, dim)

    def forward(self, origin_center, origin_offset, x_new):
        z1 = self.MLP1(origin_center)
        z2 = self.MLP2(origin_offset)
        final_input = torch.cat([z1, z2, x_new], dim=-1)
        new_offset = self.MLP3(final_input)
        new_center = self.MLP4(final_input)
        return torch.cat([new_center, new_offset, x_new], dim=-1)


class NLKOffsetIntersection(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKOffsetIntersection, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(2 * self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        center_embeddings, offset_embeddings = torch.chunk(embeddings, 2, dim=-1)  # conj*b*dim
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(offset_embeddings, dim=0)
        return offset * gate


class NLKCenterIntersection(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKCenterIntersection, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(2 * self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, z):
        center_embeddings, offset_embeddings = torch.chunk(embeddings, 2, dim=-1)
        layer2_act = self.layer2(F.relu(self.layer1(embeddings)))  # (num_conj, batch, dim)
        attention = F.softmax(z.unsqueeze(-1) * layer2_act, dim=0)  # (num_conj, batch, dim)
        embedding = torch.sum(attention * center_embeddings, dim=0)
        return embedding


class NLKDifferenceCenter(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKDifferenceCenter, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        attention = F.softmax(self.layer2(F.relu(self.layer1(emb))), dim=0)
        return torch.sum(attention * emb, dim=0)


class NLKDifferenceOffset(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(NLKDifferenceOffset, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(self.dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, loffset, z):
        all_emb = torch.cat((loffset.unsqueeze(0), z), dim=0)
        attention = F.softmax(self.layer2(F.relu(self.layer1(all_emb))), dim=0)
        return attention


class NLKEstimator(AppFOQEstimator):
    name = "NewLook"

    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, center_reg, x_reg,
                 negative_sample_size, group_number, device):
        super().__init__()
        self.name = 'newlook'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.group_number = group_number
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.cen_reg = center_reg
        self.x_reg = x_reg
        self.conj_reg = nn.Parameter(torch.Tensor([0.01]), requires_grad=False)   # TODO: this is for avoid inf
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
                                              embedding_dim=self.entity_dim)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        self.offset_embeddings = nn.Embedding(num_embeddings=n_relation, embedding_dim=self.entity_dim)
        embedding_range = torch.tensor([(self.gamma + self.epsilon) / entity_dim]).to(self.device)
        nn.init.uniform_(tensor=self.entity_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())
        nn.init.uniform_(tensor=self.offset_embeddings.weight, a=0, b=embedding_range.item())
        self.projection_net = NLKProjection(self.entity_dim, self.hidden_dim, self.group_number)
        self.intersection_offsetnet = NLKOffsetIntersection(self.entity_dim, self.hidden_dim)
        self.intersection_centernet = NLKCenterIntersection(self.entity_dim, self.hidden_dim)
        self.Difference_centernet = NLKDifferenceCenter(self.entity_dim, self.hidden_dim)
        self.Difference_offsetnet = NLKDifferenceOffset(self.entity_dim, self.hidden_dim)

        # setup group
        self.group_alignment = nn.Parameter(torch.randint(low=0, high=group_number, size=(n_entity,)),
                                            requires_grad=False)
        self.onehot_vector = nn.Parameter(torch.zeros((n_entity, group_number)).scatter_(
            dim=1, index=self.group_alignment.unsqueeze(1), value=1), requires_grad=False)
        self.relation_adjacency = nn.Parameter(torch.zeros(n_relation, group_number, group_number), requires_grad=False)

    def setup_relation_tensor(self, projections):
        for i in range(self.n_entity):
            for j in range(self.n_relation):
                for k in projections[i][j]:
                    self.relation_adjacency[j][self.group_alignment[i]][self.group_alignment[k]] = 1

    def get_entity_embedding(self, entity_ids: torch.LongTensor):
        center_emb = self.entity_embeddings(entity_ids)
        offset_emb = torch.zeros_like(center_emb).to(self.device)
        x = self.onehot_vector[entity_ids]
        return torch.cat((center_emb, offset_emb, x), dim=-1)

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        assert emb.shape[0] == len(proj_ids)
        query_center, query_offset, x_query = torch.split(emb, self.entity_dim, dim=-1)
        r_center, r_offset = self.relation_embeddings(proj_ids), self.offset_embeddings(proj_ids)
        x_new = torch.clamp(torch.matmul(x_query.unsqueeze(1), self.relation_adjacency[proj_ids]).squeeze(), 0, 1)
        final_emb = self.projection_net(query_center + r_center, r_offset, x_new)
        return final_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        box_list, x_list = [], []
        for sub_emb in conj_emb:
            sub_box, sub_x = torch.split(sub_emb, 2 * self.entity_dim, dim=-1)
            box_list.append(sub_box)
            x_list.append(sub_x)
        x_batch = torch.stack(x_list, dim=0)
        x_new = torch.prod(x_batch, dim=0)
        z = 1. / (torch.norm(F.relu(x_batch - x_new.unsqueeze(0)), p=1, dim=-1) + self.conj_reg)
        new_center = self.intersection_centernet(torch.stack(box_list, dim=0), z)
        new_offset = self.intersection_offsetnet(torch.stack(box_list, dim=0))
        return torch.cat([new_center, new_offset, x_new], dim=-1)

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        return torch.stack(disj_emb, dim=1)

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor]):
        lemb, remb = emb[0], emb[1:]
        lcenter, loffset, l_x = torch.split(lemb, self.entity_dim, dim=-1)
        center_list, offset_list, x_list = [], [], []
        for sub_emb in remb:
            sub_center, sub_offset, sub_x = torch.split(sub_emb, self.entity_dim, dim=-1)
            center_list.append(sub_center)
            offset_list.append(sub_offset)
            x_list.append(sub_x)
        rcenter, roffset = torch.stack(center_list, dim=0), torch.stack(offset_list, dim=0)  # diff*batch*group_num
        z = torch.abs(lcenter.unsqueeze(0) - rcenter) + loffset - roffset
        new_center = self.Difference_centernet(torch.cat((lcenter.unsqueeze(0), rcenter), dim=0))
        offset_attention = self.Difference_offsetnet(loffset, z)
        new_offset = torch.sum(offset_attention * torch.cat((loffset.unsqueeze(0), roffset), dim=0), dim=0)
        new_x = F.relu(l_x - torch.sum(torch.stack(x_list, dim=0), dim=0))  # TODO: This is by intuition
        return torch.cat([new_center, new_offset, new_x], dim=-1)

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    def get_negation_embedding(self, emb: torch.Tensor):
        pass

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union: bool = False):
        pred_emb = pred_emb.unsqueeze(dim=-2)
        chosen_answer, chosen_false_answer, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        positive_all_embedding = self.get_entity_embedding(torch.tensor(chosen_answer, device=self.device))
        neg_embedding = self.get_entity_embedding(torch.tensor(chosen_false_answer, device=self.device).view(-1))
        neg_embedding = neg_embedding.view(-1, self.negative_size, 2 * self.entity_dim + self.group_number)  # batch*n*dim
        if union:
            positive_union_logit = self.compute_logit(positive_all_embedding.unsqueeze(1), pred_emb)
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
            negative_union_logit = self.compute_logit(neg_embedding.unsqueeze(1), pred_emb)
            negative_logit = torch.max(negative_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(positive_all_embedding, pred_emb)
            negative_logit = self.compute_logit(neg_embedding, pred_emb)
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_logit(self, entity_emb, query_emb):
        entity_center, _, entity_x = torch.split(entity_emb, self.entity_dim, dim=-1)
        query_center_embedding, query_offset_embedding, query_x = torch.split(query_emb, self.entity_dim, dim=-1)
        delta = (entity_center - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) \
                - self.cen_reg * torch.norm(distance_in, p=1, dim=-1) \
                - self.x_reg * torch.norm(F.relu(entity_x - query_x), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False):
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)
        pred_emb = pred_emb.unsqueeze(-2)
        batch_num = find_optimal_batch(all_embedding, query_dist=pred_emb,
                                       compute_logit=self.compute_logit, union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0),
                                                pred_emb)  # b*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), pred_emb)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit
