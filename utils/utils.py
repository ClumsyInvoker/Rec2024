from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
import random

import sys


class Dataset:

    def __init__(self):
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2021)

        device = self.device

        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


class CriteoDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000):
        super(CriteoDataset, self).__init__()

        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep='\t', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        features = sparse_features + dense_features

        # 缺失值填充
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        # 连续型特征等间隔分箱
        est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        data_df[dense_features] = est.fit_transform(data_df[dense_features])

        # 离散型特征转换成连续数字，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        data_df[features] = OrdinalEncoder().fit_transform(data_df[features])

        self.data = data_df[features + ['label']].values


class MovieLensDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=1000000, task='classification'):
        super(MovieLensDataset, self).__init__()

        dtype = {
            'userId': np.int32,
            'movieId': np.int32,
            'rating': np.float16,
        }
        if read_part:
            data_df = pd.read_csv(file, sep=',', dtype=dtype, nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',', dtype=dtype)
        data_df = data_df.drop(columns=['timestamp'])

        if task == 'classification':
            data_df['rating'] = data_df.apply(lambda x: 1 if x['rating'] > 3 else 0, axis=1).astype(np.int8)

        self.data = data_df.values


class AmazonBooksDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000, sequence_length=40):
        super(AmazonBooksDataset, self).__init__()

        if read_part:
            data_df = pd.read_csv(file, sep=',', nrows=sample_num)
        else:
            data_df = pd.read_csv(file, sep=',')

        data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
        data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

        # cate encoder
        cate_list = list(data_df['cateID'])
        data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1)
        cate_set = set(cate_list + ['0'])
        cate_encoder = LabelEncoder().fit(list(cate_set))
        self.cate_set = cate_encoder.transform(list(cate_set))

        # cate pad and transform
        hist_limit = sequence_length
        col = ['hist_cate_{}'.format(i) for i in range(hist_limit)]

        def deal(x):
            if len(x) > hist_limit:
                return pd.Series(x[-hist_limit:], index=col)
            else:
                pad = hist_limit - len(x)
                x = x + ['0' for _ in range(pad)]
                return pd.Series(x, index=col)

        cate_df = data_df['hist_cate_list'].apply(deal).join(data_df[['cateID']]).apply(cate_encoder.transform).join(
            data_df['label'])
        self.data = cate_df.values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        field_dims = [self.data[:-1].max().astype(int) + 1]
        num_data = len(self.data)
        num_train = int(train_size * num_data)
        num_test = int(test_size * num_data)
        train = self.data[:num_train]
        valid = self.data[num_train: -num_test]
        test = self.data[-num_test:]

        device = self.device
        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


def create_dataset(dataset='criteo', read_part=True, sample_num=100000, task='classification', sequence_length=40, device=torch.device('cpu')):
    if dataset == 'criteo':
        return CriteoDataset('../dataset/criteo-100k.txt', read_part=read_part, sample_num=sample_num).to(device)
    elif dataset == 'movielens':
        return MovieLensDataset('../dataset/ml-latest-small-ratings.txt', read_part=read_part, sample_num=sample_num, task=task).to(device)
    elif dataset == 'amazon-books':
        return AmazonBooksDataset('../dataset/amazon-books-100k.txt', read_part=read_part, sample_num=sample_num, sequence_length=sequence_length).to(device)
    else:
        raise Exception('No such dataset!')


def evaluate(model, dataset, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_test_neg_item = args.num_test_neg_item
    for data in tqdm(dataloader, desc="Testing Progress"):
        user_id, history_items, history_items_len, \
            target_item_id, neg_item_id, user_features, item_features, neg_item_features = data

        item_idx = torch.cat([target_item_id, neg_item_id], dim=1)
        item_idx = torch.reshape(item_idx, (num_test_neg_item+1, 1))
        history_items = torch.tile(history_items, (num_test_neg_item+1, 1))
        history_items_len = torch.tile(history_items_len, (num_test_neg_item+1, 1))
        user_features = torch.tile(user_features, (num_test_neg_item+1, 1))
        item_features = item_features.unsqueeze(1)
        item_features = torch.cat([item_features, neg_item_features], dim=1)
        item_features = torch.reshape(item_features, (num_test_neg_item+1, -1))
        predictions = model(user_id, item_idx, history_items, history_items_len, user_features, item_features).squeeze(1)

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user
