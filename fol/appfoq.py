from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

IntList = List[int]
eps = 1e-6


def find_optimal_batch(answer_set: torch.tensor, query_dist: torch.tensor, compute_logit):
    batch_num = 1
    while True:
        try:
            batch_size = int(answer_set.shape[0] / batch_num)
            batch_answer_set = answer_set[0:batch_size]
            logit = compute_logit(batch_answer_set.unsqueeze(0), query_dist)
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


def compute_final_loss(positive_logit, negative_logit, subsampling_weight):
    positive_score = F.logsigmoid(positive_logit)
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
    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractmethod
    def get_disjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractmethod
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        pass

    @abstractmethod
    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]):
        pass

    @abstractmethod
    def compute_all_entity_logit(self, pred_emb: torch.Tensor):
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
    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, num_layers,
                 negative_sample_size, evaluate_union, device):
        super().__init__()
        self.name = 'beta'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.gamma = gamma
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

    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        l_alpha, l_beta = torch.chunk(lemb, 2, dim=-1)  # b*dim
        r_alpha, r_beta = torch.chunk(remb, 2, dim=-1)
        all_alpha = torch.stack([l_alpha, r_alpha])  # 2*b*dim
        all_beta = torch.stack([l_beta, r_beta])
        new_alpha, new_beta = self.intersection_net(all_alpha, all_beta)
        embedding = torch.cat([new_alpha, new_beta], dim=-1)
        return embedding

    def get_disjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        l_neg = 1. / lemb
        r_neg = 1. / remb
        neg_emb = self.get_conjunction_embedding(l_neg, r_neg)
        return 1. / neg_emb

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):  # a-b = a and(-b)
        r_neg_emb = 1. / remb
        return self.get_conjunction_embedding(lemb, r_neg_emb)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]):
        assert pred_emb.shape[0] == len(answer_set)
        alpha_embedding, beta_embedding = torch.chunk(pred_emb, 2, dim=-1)
        query_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            negative_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        answer_embedding = self.get_entity_embedding(
            torch.tensor(chosen_ans, device=self.device)).squeeze()
        positive_logit = self.compute_logit(answer_embedding, query_dist)
        all_neg_emb = self.get_entity_embedding(torch.tensor(chosen_false_ans, device=self.device).view(-1))
        all_neg_emb = all_neg_emb.view(-1, self.negative_size, 2*self.entity_dim)  # batch*negative*dim
        query_dist_unsqueezed = torch.distributions.beta.Beta(alpha_embedding.unsqueeze(1), beta_embedding.unsqueeze(1))
        negative_logit = self.compute_logit(all_neg_emb, query_dist_unsqueezed)  # b*negative
        return positive_logit, negative_logit, subsampling_weight.to(self.device)

    def compute_logit(self, entity_emb, query_dist):
        entity_alpha, entity_beta = torch.chunk(entity_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim

        pred_alpha, pred_beta = torch.chunk(pred_emb, 2, dim=-1)  # batch*dim
        query_dist = torch.distributions.beta.Beta(pred_alpha.unsqueeze(1), pred_beta.unsqueeze(1))
        batch_num = find_optimal_batch(all_embedding, query_dist=query_dist, compute_logit=self.compute_logit)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
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


class CenterIntersection(nn.Module):  # Todo: in box ,this seems to be a 2*self.dim, self.dim

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
    def __init__(self, n_entity, n_relation, gamma, entity_dim, offset_activation, center_reg, negative_size):
        super().__init__()
        self.name = 'box'
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = gamma
        self.negative_size = negative_size
        self.entity_dim = entity_dim
        self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
                                              embedding_dim=self.entity_dim)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.entity_dim)
        self.offset_embeddings = nn.Embedding(num_embeddings=n_relation, embedding_dim=self.entity_dim)
        self.offset_regularizer = Regularizer(0, 0, 1e9)
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
        center_emb = self.entity_embeddings(entity_ids.to(self.device))
        if self.use_cuda >= 0:
            offset_emb = torch.zeros_like(center_emb).to(self.device)
        else:
            offset_emb = torch.zeros_like(center_emb)
        return torch.cat((center_emb, offset_emb), dim=-1)

    def get_projection_embedding(self, proj_ids: torch.LongTensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb, r_offset_emb = self.relation_embeddings(proj_ids.to(self.device)), \
                                self.offset_embeddings(proj_ids.to(self.device))
        r_offset_emb = self.offset_regularizer(r_offset_emb)
        q_emb, q_off_emb = torch.chunk(emb, 2, dim=-1)
        q_emb = torch.add(q_emb, rel_emb)
        q_off_emb = torch.add(q_off_emb, self.func(r_offset_emb))
        return torch.cat((q_emb, q_off_emb), dim=-1)

    def get_disjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        assert False, "box cannot handle disjunction"

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        assert False, "box cannot handle negation"

    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        l_center, l_offset = torch.chunk(lemb, 2, dim=-1)
        r_center, r_offset = torch.chunk(remb, 2, dim=-1)
        new_center = self.center_net(torch.stack((l_center, r_center)))
        new_offset = self.offset_net(torch.stack((l_offset, r_offset)))
        new_offset = self.offset_regularizer(new_offset)
        return torch.cat((new_center, new_offset), dim=-1)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]) -> torch.Tensor:
        chosen_answer, chosen_false_answer, subsampling_weight = \
            negative_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        chosen_answer = torch.LongTensor(chosen_answer)
        positive_all_embedding = self.get_entity_embedding(chosen_answer.to(self.device))  # b*d
        positive_embedding, _ = torch.chunk(positive_all_embedding, 2, dim=-1)
        negative_embedding_list = []
        for i in range(len(chosen_false_answer)):
            neg_embedding = self.get_entity_embedding(torch.LongTensor(chosen_false_answer[i]).to(self.device))  # n*dim
            negative_embedding_list.append(neg_embedding)
        all_negative_embedding = torch.stack(negative_embedding_list, dim=0)  # batch*n*dim
        negative_embedding, _ = torch.chunk(all_negative_embedding, 2, dim=-1)
        positive_logit = self.compute_logit(positive_embedding, pred_emb)
        negative_logit = self.compute_logit(negative_embedding, pred_emb)
        loss = compute_final_loss(positive_logit, negative_logit, subsampling_weight.to(self.device))
        return loss

    def compute_logit(self, entity_emb, query_emb):
        query_center_embedding, query_offset_embedding = torch.chunk(query_emb, 2, dim=-1)
        delta = (entity_emb - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_entity_embedding = torch.chunk(all_entities, 2, dim=-1)
        return self.compute_logit(all_entities, pred_emb)


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
                 negative_sample_size, t_norm, bounded, use_att, use_gtrans, device, evaluate_union):
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
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.cat([lower, upper], dim=-1))
        else:
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.rand((n_entity, self.entity_dim * 2)))
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        embedding_range = torch.tensor([(self.gamma + self.epsilon) / hidden_dim]).to(self.device)
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())

        self.center_net = LogicIntersection(self.entity_dim, t_norm, bounded, use_att, use_gtrans)
        self.projection_net = LogicProjection(self.entity_dim * 2, self.relation_dim, hidden_dim, num_layers, bounded)

    def get_entity_embedding(self, entity_ids: torch.Tensor):
        emb = self.entity_embeddings(entity_ids.to(self.device))
        return emb

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = self.relation_embeddings(proj_ids.to(self.device))
        pro_emb = self.projection_net(emb, rel_emb)
        return pro_emb

    def get_negation_embedding(self, embedding: torch.Tensor):
        if self.bounded:
            lower_embedding, upper_embedding = torch.chunk(embedding, 2, dim=-1)
            embedding = torch.cat([1 - upper_embedding, 1 - lower_embedding], dim=-1)
        else:
            embedding = 1 - embedding
        return embedding

    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        all_emb = torch.stack([lemb, remb])
        emb = self.center_net(all_emb)
        return emb

    def get_disjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        n_lemb = self.get_negation_embedding(lemb)
        n_remb = self.get_negation_embedding(remb)
        n_emb = self.get_conjunction_embedding(n_lemb, n_remb)
        emb = self.get_negation_embedding(n_emb)
        return emb

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        n_remb = self.get_negation_embedding(remb)
        return self.get_conjunction_embedding(lemb, n_remb)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]):
        assert pred_emb.shape[0] == len(answer_set)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            negative_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        answer_embedding = self.get_entity_embedding(
            torch.tensor(chosen_ans, device=self.device)).squeeze()
        positive_logit = self.compute_logit(answer_embedding, pred_emb)
        negative_embedding_list = []
        for i in range(len(chosen_false_ans)):  # todo: is there a way to parallelize
            neg_embedding = self.get_entity_embedding(torch.tensor(chosen_false_ans[i], device=self.device))  # n*dim
            negative_embedding_list.append(neg_embedding)
        all_negative_embedding = torch.stack(negative_embedding_list, dim=0)  # batch*negative*dim
        negative_logit = self.compute_logit(all_negative_embedding, pred_emb.unsqueeze(dim=1))  # b*negative
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

    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        query_entropy = None
        if self.bounded:
            lower, upper = torch.chunk(pred_emb, 2, dim=-1)
            truth_interval = upper - lower
            distribution = torch.distributions.uniform.Uniform(lower, upper + eps)
            query_entropy = (distribution.entropy(), truth_interval)
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        batch_num = find_optimal_batch(all_embedding, query_dist=pred_emb, compute_logit=self.compute_logit)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            logit_part = self.compute_logit(answer_part.unsqueeze(dim=0), pred_emb)  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit, query_entropy


class CQDEstimator(AppFOQEstimator):
    def __init__(self, n_entity, n_relation, hidden_dim,
                 gamma, entity_dim, relation_dim, num_layers,
                 negative_sample_size, device, evaluate_union):
        super().__init__()
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        embedding_range = torch.tensor([(self.gamma + self.epsilon) / hidden_dim]).to(self.device)
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-embedding_range.item(), b=embedding_range.item())









