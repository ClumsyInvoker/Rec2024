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


class MyDataset(Dataset):
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
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = []
            for idx in range(len(self.data)):
                user_id = int(self.data.iloc[idx]['user_id'])
                neg_item_ids = []
                if self.mode == 'val':
                    total_history_items = self.user_history_positive[user_id][:-1]
                else:
                    total_history_items = self.user_history_positive[user_id]
                for _ in range(self.neg_num):
                    neg_item_id = random_neq(0, self.item_num, set(total_history_items + neg_item_ids))
                    neg_item_ids.append(neg_item_id)
                self.data_neg_items.append(neg_item_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        row = self.data.iloc[idx]
        user_id = int(row['user_id'])
        target_item_id = int(row['item_id'])
        positive_behavior_offset = int(row['positive_behavior_offset'])
        label = row['label']
        cold_item = row['cold_item']

        raw_history_items = self.user_history_positive[user_id][
                        max(0, positive_behavior_offset + 1 - self.max_length):positive_behavior_offset + 1]
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]

        if self.mode == 'train':
            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)
            cold_item = torch.FloatTensor([cold_item]).to(self.device)

            return user_id, history_items, history_items_len, target_item_id,\
                user_features, item_features, label, cold_item

        elif self.mode == 'val' or self.mode == 'test':
            # neg_item_ids = []
            # if self.mode == 'val':
            #     total_history_items = self.user_history_positive[user_id][:-1]
            # else:
            #     total_history_items = self.user_history_positive[user_id]
            # for _ in range(self.neg_num):
            #     neg_item_id = random_neq(0, self.item_num, set(total_history_items + neg_item_ids))
            #     neg_item_ids.append(neg_item_id)
            neg_item_ids = self.data_neg_items[idx]
            neg_item_features = self.item_features.iloc[neg_item_ids]

            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features

        else:
            raise ValueError('mode must be train/val/test')


class PTCRDataset(Dataset):
    def __init__(self, data_dir, max_length=50, feedback_max_length=10, mode='train', neg_num=1, device='cpu'):
        self.mode = mode
        self.max_length = max_length
        self.feedback_max_length = feedback_max_length
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
        self.data = pd.read_csv(data_dir + '/' + mode + '_data.csv')
        # self.user_history = pd.read_pickle(data_dir + '/user_history.pkl')
        # self.item_history = pd.read_pickle(data_dir + '/item_history.pkl')
        self.user_history_positive = pd.read_pickle(data_dir + '/user_history_positive.pkl')
        self.user_history_negative = pd.read_pickle(data_dir + '/user_history_negative.pkl')
        self.item_history_positive = pd.read_pickle(data_dir + '/item_history_positive.pkl')
        self.item_history_negative = pd.read_pickle(data_dir + '/item_history_negative.pkl')

        # 如果是test或val，则为每个样本生成negative item, 固定为100个
        if self.mode == 'val' or self.mode == 'test':
            self.data_neg_items = []
            self.data_neg_item_pos_feedbacks = []
            self.data_neg_item_pos_feedback_lens = []
            for idx in range(len(self.data)):
                user_id = int(self.data.iloc[idx]['user_id'])
                neg_item_ids = []
                neg_item_pos_feedbacks = []
                neg_item_pos_feedback_lens = []
                if self.mode == 'val':
                    total_history_items = self.user_history_positive[user_id][:-1]
                else:
                    total_history_items = self.user_history_positive[user_id]
                for _ in range(self.neg_num):
                    neg_item_id = random_neq(0, self.item_num, set(total_history_items + neg_item_ids))
                    raw_neg_item_pos_feedback = self.item_history_positive[neg_item_id][-self.feedback_max_length:]
                    neg_item_pos_feedback = torch.LongTensor(
                        raw_neg_item_pos_feedback + [0] * (self.feedback_max_length - len(raw_neg_item_pos_feedback)))
                    neg_item_pos_feedback_len = torch.LongTensor(
                        [1] * len(raw_neg_item_pos_feedback) + [0] * (
                                    self.feedback_max_length - len(raw_neg_item_pos_feedback)))
                    neg_item_ids.append(neg_item_id)
                    neg_item_pos_feedbacks.append(neg_item_pos_feedback)
                    neg_item_pos_feedback_lens.append(neg_item_pos_feedback_len)
                self.data_neg_items.append(neg_item_ids)
                self.data_neg_item_pos_feedbacks.append(neg_item_pos_feedbacks)
                self.data_neg_item_pos_feedback_lens.append(neg_item_pos_feedback_lens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取指定索引处的数据
        row = self.data.iloc[idx]
        user_id = int(row['user_id'])
        target_item_id = int(row['item_id'])
        positive_behavior_offset = int(row['positive_behavior_offset'])
        item_positive_behavior_offset = int(row['item_positive_behavior_offset'])
        item_negative_behavior_offset = int(row['item_negative_behavior_offset'])
        label = row['label']
        cold_item = row['cold_item']
        raw_history_items = self.user_history_positive[user_id][
                        max(0, positive_behavior_offset + 1 - self.max_length):positive_behavior_offset + 1]
        user_features = self.user_features.iloc[user_id]
        item_features = self.item_features.iloc[target_item_id]

        if self.mode == 'train':
            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)

            raw_item_pos_feedback = self.item_history_positive[target_item_id][max(0, item_positive_behavior_offset + 1 -
                                                                      self.feedback_max_length):item_positive_behavior_offset + 1]
            raw_item_neg_feedback = self.item_history_negative[target_item_id][max(0, item_negative_behavior_offset + 1 -
                                                                      self.feedback_max_length):item_negative_behavior_offset + 1]

            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)
            cold_item = torch.FloatTensor([cold_item]).to(self.device)

            item_pos_feedback = torch.LongTensor(raw_item_pos_feedback + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))).to(self.device)
            item_pos_feedback_len = torch.LongTensor([1] * len(raw_item_pos_feedback) + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))).to(self.device)
            item_neg_feedback = torch.LongTensor(raw_item_neg_feedback + [0] * (self.feedback_max_length - len(raw_item_neg_feedback))).to(self.device)
            item_neg_feedback_len = torch.LongTensor([1] * len(raw_item_neg_feedback) + [0] * (self.feedback_max_length - len(raw_item_neg_feedback))).to(self.device)

            return user_id, history_items, history_items_len, target_item_id,\
                user_features, item_features, label, cold_item, \
                item_pos_feedback, item_pos_feedback_len, item_neg_feedback, item_neg_feedback_len

        elif self.mode == 'val' or self.mode == 'test':
            raw_item_pos_feedback = self.item_history_positive[target_item_id][
                                    max(0, item_positive_behavior_offset + 1 -
                                        self.feedback_max_length):item_positive_behavior_offset + 1]
            item_pos_feedback = torch.LongTensor(
                raw_item_pos_feedback + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))).to(self.device)
            item_pos_feedback_len = torch.LongTensor(
                [1] * len(raw_item_pos_feedback) + [0] * (self.feedback_max_length - len(raw_item_pos_feedback))).to(
                self.device)

            # neg_item_ids = []
            # neg_item_pos_feedbacks = []
            # neg_item_pos_feedback_lens = []
            # if self.mode == 'val':
            #     total_history_items = self.user_history_positive[user_id][:-1]
            # else:
            #     total_history_items = self.user_history_positive[user_id]
            # for _ in range(self.neg_num):
            #     neg_item_id = random_neq(0, self.item_num, set(total_history_items + neg_item_ids))
            #     raw_neg_item_pos_feedback = self.item_history_positive[neg_item_id][-self.feedback_max_length:]
            #     neg_item_pos_feedback = torch.LongTensor(
            #         raw_neg_item_pos_feedback + [0] * (self.feedback_max_length - len(raw_neg_item_pos_feedback)))
            #     neg_item_pos_feedback_len = torch.LongTensor(
            #         [1] * len(raw_neg_item_pos_feedback) + [0] * (self.feedback_max_length - len(raw_neg_item_pos_feedback)))
            #     neg_item_ids.append(neg_item_id)
            #     neg_item_pos_feedbacks.append(neg_item_pos_feedback)
            #     neg_item_pos_feedback_lens.append(neg_item_pos_feedback_len)

            neg_item_ids = self.data_neg_items[idx]
            neg_item_pos_feedbacks = self.data_neg_item_pos_feedbacks[idx]
            neg_item_pos_feedback_lens = self.data_neg_item_pos_feedback_lens[idx]

            neg_item_features = self.item_features.iloc[neg_item_ids]
            neg_item_pos_feedbacks = torch.stack(neg_item_pos_feedbacks, dim=0).to(
                self.device) # [neg_num, feedback_max_length]
            neg_item_pos_feedback_lens = torch.stack(neg_item_pos_feedback_lens, dim=0).to(
                self.device) # [neg_num, feedback_max_length]

            # 转化成tensor
            user_id = torch.LongTensor([user_id]).to(self.device)
            history_items = torch.LongTensor(raw_history_items + [0] * (self.max_length - len(raw_history_items))).to(
                self.device)
            history_items_len = torch.LongTensor(
                [1] * len(raw_history_items) + [0] * (self.max_length - len(raw_history_items))).to(self.device)
            target_item_id = torch.LongTensor([target_item_id]).to(self.device)
            neg_item_id = torch.LongTensor(neg_item_ids).to(self.device)
            user_features = torch.FloatTensor(user_features.values).to(self.device)
            item_features = torch.FloatTensor(item_features.values).to(self.device)
            neg_item_features = torch.FloatTensor(neg_item_features.values).to(self.device)

            # 返回样本
            return user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features, \
                   item_pos_feedback, item_pos_feedback_len, neg_item_pos_feedbacks, neg_item_pos_feedback_lens

        else:
            raise ValueError('mode must be train/val/test')
