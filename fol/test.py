import torch

batch = 2
nega = 4
dim = 2

easy_ans = [[0], []]
hard_ans = [[2, 3], [1]]

a = torch.randn(batch, nega)
c = 100000
print(c // 2)
print(a, 1 / a, 1./a, a.sum())

