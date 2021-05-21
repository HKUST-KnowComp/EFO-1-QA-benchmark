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




print(a[0])

print(b(idx))

