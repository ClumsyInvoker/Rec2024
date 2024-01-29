import torch

a = torch.Tensor([8, 9, 1, 3, 5])
print(a.argsort().argsort(descending=True))