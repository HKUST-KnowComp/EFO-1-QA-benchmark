import torch

batch = 2
nega = 4
dim = 2

easy_ans = [[0], []]
hard_ans = [[2, 3], [1]]

a = torch.randn(batch, nega)
sorted_a,  argsort = torch.sort(a, dim=1, descending=True)
ranking = argsort.clone().to(torch.float)
ranking = ranking.scatter_(
                        1, argsort, torch.arange(nega).to(torch.float).repeat(argsort.shape[0], 1))

print(a)
print(sorted_a, argsort)
print(ranking)

for i in range(ranking.shape[0]):
    cur_ranking = ranking[i][easy_ans[i]+hard_ans[i]]
    print(cur_ranking)
    cur_ranking, indices = torch.sort(cur_ranking)
    print(cur_ranking)
    num_easy, num_hard = len(easy_ans[i]), len(hard_ans[i])
    masks = indices >= num_easy
    answer_list = torch.arange(
        num_hard + num_easy).to(torch.float)
    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
    print(cur_ranking)
    # only take indices that belong to the hard answers
    cur_ranking = cur_ranking[masks]
    print(cur_ranking)
    mrr = torch.mean(1. / cur_ranking).item()
    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
    h10 = torch.mean(
        (cur_ranking <= 10).to(torch.float)).item()

