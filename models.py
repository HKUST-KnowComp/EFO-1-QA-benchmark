#!/usr/bin/python3

from __future__ import absolute_import, division, print_function

import collections
import itertools
import logging
import math
import os
import pickle
import random
import time  #ddd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import (SingledirectionalOneShotIterator, TestDataset,
                        TrainDataset)


def Identity(x):
    return x


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


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class DiscreteMeasure_ProjectionNet(nn.Module):  # 如果MLP的话没区别
    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers, projection_regularizer):
        super(DiscreteMeasure_ProjectionNet, self).__init__()
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


class DiscreteMeasure_GaussianNet(nn.Module):   # to be finished
    def __init__(self,
                 entity_dim,
                 relation_dim,
                 hidden_dim,
                 support_location,
                 projection_regularizer,
                 batch_size,
                 k):

        super(DiscreteMeasure_GaussianNet, self).__init__()
        self.support_num = entity_dim
        self.support_dim = support_location.size()[1]
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.support_location = support_location
        self.projection_regulazer = projection_regularizer
        self.batch_size = batch_size
        self.k = k  # k as the top_k
        self.layer1 = nn.Linear(
            self.relation_dim+self.support_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.support_dim)
        # self.layer3=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, e_embedding, r_embedding):
        weight = e_embedding  # batch*support_num
        mean = self.support_location  # support_num*support_dim
        # mean=self.layer1(mean)  #support_num*hidden_dim
        # r_embedding=self.layer2(r_embedding)  #batch*support_dim -> batch*hidden_dim
        mean = self.MLP(mean, r_embedding)  # batch*support_num*support_dim
        top_weight, indices = torch.topk(weight, self.k)
        indices = indices.unsqueeze(2).repeat(1, 1, self.support_dim)
        top_mean = torch.gather(mean, dim=1, index=indices)  # [b,k,dim]
        F = self.support_location.unsqueeze(1).unsqueeze(
            0) - top_mean.unsqueeze(1)  # [b,num,k,dim]
        prob = torch.norm(F, dim=3)  # [b,num,k]
        prob = ((-1 / 2) * (torch.pow(prob, 2))).exp()
        top_weight = top_weight.unsqueeze(1)  # [b,1,k]
        new_weight = torch.sum(torch.mul(top_weight, prob), dim=-1)
        new_weight = self.projection_regulazer(new_weight)
        return new_weight

    def MLP(self, Gaussian_mean, r_embedding):
        new_Gaussian_mean = Gaussian_mean.unsqueeze(0).repeat(
            r_embedding.size()[0], 1, 1)  # [1,num,support_dim]
        r_embedding = r_embedding.unsqueeze(1).repeat(
            1, Gaussian_mean.size()[0], 1)  # [b,1,relation_dim]
        new_support_location = F.relu(self.layer1(
            torch.cat((r_embedding, new_Gaussian_mean), 2)))
        new_support_location = self.layer2(new_support_location)
        return new_support_location


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, batch_size=512, GMM_mode=None, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, discrete_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        if use_cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(use_cuda))
        else:
            self.device = torch.device('cpu')
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).to(
            self.device) if self.use_cuda >= 0 else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict
        self.batch_size = batch_size
        self.use_Gaussian, self.topk = GMM_mode
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.embedding_low_bound = -self.embedding_range.item()
        self.embedding_up_bound = self.embedding_range.item()

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(
                nentity, self.entity_dim))  # centor for entities
            activation, cen = box_mode
            self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus

        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(
                nentity, self.entity_dim))  # center for entities
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(
                nentity, self.entity_dim * 2))  # alpha and beta
            # make sure the parameters of beta embeddings are positive
            self.entity_regularizer = Regularizer(1, 0.05, 1e9)
            # make sure the parameters of beta embeddings after relation projection are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        elif self.geo == 'discrete':
            self.support_num = hidden_dim
            self.support_dim, self.relation_dim, projection_hidden_dim, self.num_layers = discrete_mode
            self.entity_embedding = nn.Parameter(
                torch.zeros(nentity, self.support_num))
            self.support_embedding = nn.Parameter(torch.zeros(
                self.support_num, self.support_dim), requires_grad=False)
            nn.init.uniform_(tensor=self.support_embedding, a=-1, b=1)
            self.embedding_low_bound = 0
            self.embedding_up_bound = 1
            self.entity_regularizer = Regularizer(0, 0, 1)
            self.projection_regularizer = Regularizer(0, 0, 1)

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_low_bound,
            b=self.embedding_up_bound
        )
        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_low_bound,
            b=self.embedding_up_bound
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(
                torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)
        elif self.geo == 'discrete':
            if self.use_Gaussian:
                self.projection_net = DiscreteMeasure_GaussianNet(entity_dim=self.entity_dim, relation_dim=self.relation_dim,
                                                                  hidden_dim=projection_hidden_dim, support_location=self.support_embedding,
                                                                  projection_regularizer=self.projection_regularizer, batch_size=self.batch_size, k=self.topk)
            else:
                self.projection_net = DiscreteMeasure_ProjectionNet(entity_dim=self.entity_dim,
                                                                    relation_dim=self.relation_dim, hidden_dim=projection_hidden_dim,
                                                                    num_layers=self.num_layers,
                                                                    projection_regularizer=self.projection_regularizer)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'discrete':
            return self.forward_discrete(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_discrete(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_discrete(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                           query_structure),
                                                                self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_discrete(
                    batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(
                all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(
                all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(
                all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_discrete(
                    positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_discrete(
                    positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_discrete(
                    negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_discrete(
                    negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_discrete(self, entity_embedding, query_embedding):
        score = torch.sum(torch.mul(entity_embedding, query_embedding), dim=-1)
        logit = score-self.gamma
        return logit

    def embed_query_discrete(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                discrete_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                discrete_embedding, idx = self.embed_query_discrete(
                    queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    discrete_embedding = 1-discrete_embedding
                else:  # 只可能是r
                    r_embedding = torch.index_select(
                        self.relation_embedding, dim=0, index=queries[:, idx])
                    discrete_embedding = self.projection_net(
                        discrete_embedding, r_embedding)
                idx += 1
        else:  # 最后一个是i或者更复杂
            discrete_embedding = torch.ones(self.entity_dim).to(self.device)
            for i in range(len(query_structure)):
                new_discrete_embedding, idx = self.embed_query_discrete(
                    queries, query_structure[i], idx)
                discrete_embedding = torch.min(
                    discrete_embedding, new_discrete_embedding)

        return discrete_embedding, idx

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda >= 0:
                    offset_embedding = torch.zeros_like(
                        embedding).to(self.device)
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(
                    queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(
                        self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(
                        self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(
                    queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(
                torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(
                    queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(
                        self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(
                    queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        # print("queries",queries,queries.size(),query_structure,idx)
        all_relation_flag = True
        # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(
                    queries, query_structure[0], idx)
                embedding = torch.cat(
                    [alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(
                        self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                    """
                        self.entity_embedding = [0, 1]^{#entity \times #support}
                        embedding \in [0, 1]^{#support}, row of self.entity_embedding
                        self.relation_embedding = R^{#relation \time dim(support)}
                        r_embedding \in R^{dim(support)}
                        self.projection_net(embedding, r_embedding)
                        -- ProjectNetMeasure
                            def __init__(self, num_support, dim_support):
                                self.support = matrix(num_support, dim_support)

                            def forward(embedding, r_embedding)
                                1. discrete to Gaussian
                                    input: embedding; self.support
                                    output: mixture Gaussian
                                2. Gaussian TranE/RotatE
                                    input: mixture Gaussian
                                    output: shift mean $\mu + r_embedding$ by r_embedding
                                3. Gaussian to discrete
                                    new embedding

                            def support_update(self, entity_embedding):
                                1. identify useless support, i.e. k-th row in self.support
                                2. k-th col of entity_embedding set to be zero?
                                3. ...
                    """
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(
                    queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(
                torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))
        # print('answers',alpha_embedding.size())
        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(
            entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(
            alpha_embedding, beta_embedding)
        logit = self.gamma - \
            torch.norm(torch.distributions.kl.kl_divergence(
                entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(
                                              query_structure),
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(
                all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(
                all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(
                all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(
                all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(
                all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(
                all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(
                all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(
                all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(
                    positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(
                    positive_embedding, all_union_dists)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(
                    negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(
                    negative_embedding, all_union_dists)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat(
                [queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - \
            torch.norm(distance_out, p=1, dim=-1) - self.cen * \
            torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(
                                             query_structure),
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(
                all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(
                all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(
                all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(
                all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(
                all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(
                all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(
                    positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(
                    positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(
                    negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(
                    negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                      query_structure),
                                                           self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(
                    batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(
                all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(
                all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(
                all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(
                    positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(
                    positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(
                    negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(
                    negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, device):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(
            train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        # group queries with same structure
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure]).to(device)
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)

        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, device, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).to(device)
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.to(device)

                _, negative_logit, _, idxs = model(
                    None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                if len(argsort) == args.test_batch_size:
                    # achieve the ranking of all entities
                    ranking = ranking.scatter_(
                        1, argsort, model.batch_entity_range)
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).to(device)
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(
                        easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).to(device)
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    # only take indices that belong to the hard answers
                    cur_ranking = cur_ranking[masks]

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' %
                                 (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum(
                    [log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(
                logs[query_structure])

        return metrics
