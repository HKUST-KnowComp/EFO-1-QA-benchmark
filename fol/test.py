import torch
import torch.nn as nn
import numpy as np
import random
batch = 2
nega = 4
dim = 2

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


torch.backends.cudnn.deterministic = True
easy_ans = [[0], []]
hard_ans = [[2, 3], [1]]

a = nn.Parameter(torch.zeros(batch, dim))
b = nn.Embedding(num_embeddings=batch, embedding_dim=dim)
idx = torch.tensor(0)
embed_range = 62/500
nn.init.uniform_(tensor=a, a=-embed_range, b=embed_range)
nn.init.uniform_(tensor=b.weight, a=-embed_range, b=embed_range)


box_path = 'data/FB15k-237-q2b'
beta_path = 'data/FB15k-237-betae'

import pickle, os


def load(path):
    valid_queries = pickle.load(open(os.path.join(path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(path, "test-easy-answers.pkl"), 'rb'))
    return valid_easy_answers, valid_hard_answers, test_easy_answers, test_hard_answers, valid_queries, test_queries


box_valid_e, box_valid_h, box_test_e, box_test_h, box_valid, box_test = load(box_path)
beta_valid_e, beta_valid_h, beta_test_e, beta_test_h, beta_valid, beta_test = load(beta_path)


for q_s in box_valid.keys():
    box_all_valid_q, box_all_test_q = box_valid[q_s], box_test[q_s]
    beta_all_valid_q, beta_all_test_q = beta_valid[q_s], beta_test[q_s]
    box_valid_left = box_all_valid_q - beta_all_valid_q
    box_test_left = box_all_test_q - beta_all_test_q
    for q in box_valid_left:
        num_hard, num_easy = len(box_valid_h[q]), len(box_valid_e[q])
        print(num_hard, num_easy)
    for q in box_test_left:
        num_hard, num_easy = len(box_test_h[q]), len(box_test_e[q])
        print(num_hard, num_easy)

