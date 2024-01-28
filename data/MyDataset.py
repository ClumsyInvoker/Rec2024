import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class SequentialRecommendationDataset(Dataset):
    def __init__(self, data_dir, max_length=50, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.neg_num = neg_num
        self.device = device

        # 从CSV文件中加载meta data
        meta_data = pd.read_csv(data_dir + '/meta_data.csv')
        self.name = meta_data['dataset_name'].values[0]
        self.user_num = meta_data['user_num'].values[0]
        self.item_num = meta_data['item_num'].values[0]

        # 加载用户和物品的特征
        self.user_features = pd.read_csv(data_dir + '/user_features.csv')
        self.user_features_dim = self.user_features.shape[1]
        self.item_features = pd.read_csv(data_dir + '/item_features.csv')
        self.item_features_dim = self.item_features.shape[1]

        # 加载交互数据
        raw_data = pd.read_csv(data_dir + '/data.csv')
        self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.data = range(self.user_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        user_id = self.data[idx]
        if self.mode == 'train':
            target_item_id = self.user_history[user_id][-3]
            history_items = self.user_history[user_id][-(self.max_length+3):-3]
        elif self.mode == 'val':
            target_item_id = self.user_history[user_id][-2]
            history_items = self.user_history[user_id][-(self.max_length+2):-2]
        elif self.mode == 'test':
            target_item_id = self.user_history[user_id][-1]
            history_items = self.user_history[user_id][-(self.max_length+1):-1]
        else:
            raise ValueError('mode must be train/val/test')

        neg_item_ids = []
        for _ in range(self.neg_num):
            neg_item_id = random_neq(0, self.item_num, set(history_items+[target_item_id]+neg_item_ids))
            neg_item_ids.append(neg_item_id)

        # 获取用户和物品的特征
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]
        neg_item_features = self.item_features.iloc[neg_item_ids]

        # 转化成tensor
        user_id = torch.LongTensor([user_id]).to(self.device)
        history_items = torch.LongTensor(history_items + [0] * (self.max_length-len(history_items))).to(self.device)
        history_items_len = torch.LongTensor([1] * len(history_items) + [0] * (self.max_length-len(history_items))).to(self.device)
        target_item_id = torch.LongTensor([target_item_id]).to(self.device)
        neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
        user_features = torch.FloatTensor(user_features.values).to(self.device)
        item_features = torch.FloatTensor(item_features.values).to(self.device)
        neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

        # 返回样本
        return user_id, history_items, history_items_len, \
            target_item_id, neg_item_id, \
            user_features, item_features, neg_item_features
