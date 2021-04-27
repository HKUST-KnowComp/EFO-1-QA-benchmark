from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

IntList = List[int]


def negative_sampling(answer_set: List[IntList], negative_size: int, entity_num: int, k=1):
    all_chosen_ans = []
    all_chosen_false_ans = []
    subsampling_weight = torch.zeros(len(answer_set))
    for i in range(len(answer_set)):
        all_chosen_ans.append(random.choices(answer_set[i], k=k))
        subsampling_weight[i] = len(answer_set[i])
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
    positive_loss = -(positive_score * subsampling_weight)
    negative_loss = -(negative_score * subsampling_weight)
    loss = (positive_loss + negative_loss) / 2
    loss /= subsampling_weight.sum()
    loss = loss.sum()
    return loss


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
    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, n_entity, n_relation, hidden_dim, gamma, device,
                 entity_dim, relation_dim, num_layers, negative_sample_size, evaluate_union):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.device = device
        self.gamma = gamma
        self.negative_size = negative_sample_size
        self.entity_dim, self.relation_dim = entity_dim, relation_dim
        self.entity_embeddings = nn.Embedding(num_embeddings=n_entity,
                                              embedding_dim=self.entity_dim * 2)
        self.relation_embeddings = nn.Embedding(num_embeddings=n_relation,
                                                embedding_dim=self.relation_dim)
        self.entity_regularizer = Regularizer(0, 0.05, 1e9)
        self.projection_regularizer = Regularizer(0, 0.05, 1e9)
        self.intersection_net = BetaIntersection(self.entity_dim)
        self.projection_net = BetaProjection(self.entity_dim * 2,
                                             self.relation_dim,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)

    def get_entity_embedding(self, entity_ids: torch.IntTensor):
        emb = self.entity_embeddings(entity_ids).to(self.device)
        return self.entity_regularizer(emb)

    def get_projection_embedding(self, proj_ids: torch.IntTensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb = self.entity_regularizer(self.relation_embeddings(proj_ids))
        pro_emb = self.projection_net(emb, rel_emb)
        return pro_emb

    def get_conjunction_embedding(self, lemb: torch.Tensor, remb: torch.Tensor):
        l_alpha, l_beta = torch.chunk(lemb, 2, dim=-1)  # b*dim
        r_alpha, r_beta = torch.chunk(remb, 2, dim=-1)
        all_alpha = torch.stack([l_alpha, r_alpha])  # 2*b*dim
        all_beta = torch.stack([l_alpha, r_beta])
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
        return self.get_disjunction_embedding(lemb, r_neg_emb)

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList]) -> torch.Tensor:
        alpha_embedding, beta_embedding = torch.chunk(pred_emb, 2, dim=-1)
        query_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            negative_sampling(answer_set, negative_size=self.negative_size, entity_num=self.n_entity)
        answer_embedding = self.get_entity_embedding(torch.IntTensor(chosen_ans))  # todo : fix this when cuda>=0
        positive_logit = self.compute_logit(answer_embedding, query_dist)
        negative_embedding_list = []
        for i in range(len(chosen_false_ans)):  # todo: is there a way to parallelize
            neg_embedding = self.get_entity_embedding(torch.IntTensor(chosen_false_ans[i]))  # n*dim
            negative_embedding_list.append(neg_embedding)
        all_negative_embedding = torch.stack(negative_embedding_list, dim=0)  # batch*negative*dim
        query_dist_unsqueezed = torch.distributions.beta.Beta(alpha_embedding.unsqueeze(1), beta_embedding.unsqueeze(1))
        negative_logit = self.compute_logit(all_negative_embedding, query_dist_unsqueezed)  # b*negative
        loss = compute_final_loss(positive_logit, negative_logit, subsampling_weight)
        return loss

    def compute_logit(self, entity_emb, query_dist):
        entity_alpha, entity_beta = torch.chunk(entity_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(entity_alpha, entity_beta)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> torch.Tensor:
        all_entities = torch.IntTensor(range(self.n_entity))
        if self.device != torch.device('cpu'):
            all_entities = all_entities.to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        all_embedding = all_embedding.unsqueeze(dim=0)  # 1*nentity*dim
        pred_alpha, pred_beta = torch.chunk(pred_emb, 2, dim=-1)  # batch*dim
        query_dist = torch.distributions.beta.Beta(pred_alpha.unsqueeze(1), pred_beta.unsqueeze(1))
        return self.compute_logit(all_embedding, query_dist)


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
    def __init__(self, n_entity, n_relation, gamma, device, entity_dim, offset_activation, center_reg, negative_size):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.device = device
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

    def get_entity_embedding(self, entity_ids: torch.IntTensor):
        center_emb = self.entity_embeddings(entity_ids)
        if self.use_cuda >= 0:
            offset_emb = torch.zeros_like(center_emb).to(self.device)
        else:
            offset_emb = torch.zeros_like(center_emb)
        return torch.cat((center_emb, offset_emb), dim=-1)

    def get_projection_embedding(self, proj_ids: torch.IntTensor, emb):
        assert emb.shape[0] == len(proj_ids)
        rel_emb, r_offset_emb = self.relation_embeddings(proj_ids), self.offset_embeddings(proj_ids)
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
        chosen_answer = torch.IntTensor(chosen_answer)
        positive_all_embedding = self.get_entity_embedding(chosen_answer)  # b*d
        positive_embedding, _ = torch.chunk(positive_all_embedding, 2, dim=-1)
        negative_embedding_list = []
        for i in range(len(chosen_false_answer)):
            neg_embedding = self.get_entity_embedding(torch.IntTensor(chosen_false_answer[i]))  # n*dim
            negative_embedding_list.append(neg_embedding)
        all_negative_embedding = torch.stack(negative_embedding_list, dim=0)  # batch*n*dim
        negative_embedding, _ = torch.chunk(all_negative_embedding, 2, dim=-1)
        positive_logit = self.compute_logit(positive_embedding, pred_emb)
        negative_logit = self.compute_logit(negative_embedding, pred_emb)
        loss = compute_final_loss(positive_logit, negative_logit, subsampling_weight)
        return loss

    def compute_logit(self, entity_emb, query_emb):
        query_center_embedding, query_offset_embedding = torch.chunk(query_emb, 2, dim=-1)
        delta = (entity_emb - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def compute_all_entity_logit(self, pred_emb: torch.Tensor) -> torch.Tensor:
        all_entities = torch.IntTensor(range(self.n_entity))
        if self.device != torch.device('cpu'):
            all_entities = all_entities.to(self.device)
        all_entity_embedding = torch.chunk(all_entities, 2, dim=-1)
        return self.compute_logit(all_entities, pred_emb)