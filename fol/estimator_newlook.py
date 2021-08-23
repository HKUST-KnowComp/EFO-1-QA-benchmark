from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)


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
        return self.get_multiple_difference_embedding([lemb, remb])

    def get_negation_embedding(self, emb: torch.Tensor):
        assert False, "NewLook cannot handle negation"

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
