import torch

# 创建示例 logits，labels 和 cold_items
logits = torch.tensor([0.1, 0.5, 0.8, 0.2], dtype=torch.float32)
labels = torch.tensor([0, 1, 1, 0], dtype=torch.int)
cold_items = torch.tensor([1, 0, 1, 0], dtype=torch.int)

# 筛选出 label 为 1 且 cold_item 为 0 的 logits
selected_logits = logits[(labels == 1) & (cold_items == 0)]

# 打印结果
print(selected_logits)