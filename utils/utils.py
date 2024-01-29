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

# def create_dataset(dataset='criteo', read_part=True, sample_num=100000, task='classification', sequence_length=40, device=torch.device('cpu')):
#     if dataset == 'criteo':
#         return CriteoDataset('../dataset/criteo-100k.txt', read_part=read_part, sample_num=sample_num).to(device)
#     elif dataset == 'movielens':
#         return MovieLensDataset('../dataset/ml-latest-small-ratings.txt', read_part=read_part, sample_num=sample_num, task=task).to(device)
#     elif dataset == 'amazon-books':
#         return AmazonBooksDataset('../dataset/amazon-books-100k.txt', read_part=read_part, sample_num=sample_num, sequence_length=sequence_length).to(device)
#     else:
#         raise Exception('No such dataset!')


def evaluate(model, dataset, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_test_neg_item = args.num_test_neg_item
    for data in tqdm(dataloader, desc="Testing Progress"):
        user_id, history_items, history_items_len, \
            target_item_id, neg_item_id, user_features, item_features, neg_item_features = data

        user_id = torch.tile(user_id, (num_test_neg_item+1, 1))
        item_idx = torch.cat([target_item_id, neg_item_id], dim=1)
        item_idx = torch.reshape(item_idx, (num_test_neg_item+1, 1))
        history_items = torch.tile(history_items, (num_test_neg_item+1, 1))
        history_items_len = torch.tile(history_items_len, (num_test_neg_item+1, 1))
        user_features = torch.tile(user_features, (num_test_neg_item+1, 1))
        item_features = item_features.unsqueeze(1)
        item_features = torch.cat([item_features, neg_item_features], dim=1)
        item_features = torch.reshape(item_features, (num_test_neg_item+1, -1))
        predictions = model(user_id, item_idx, history_items, history_items_len, user_features, item_features).squeeze(1)

        rank = predictions.argsort(descending=True).argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

def evaluate_prompt(model, dataset, args, mode='test'):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_test_neg_item = args.num_test_neg_item
    desc = "Testing Progress" if mode == 'test' else "Validating Progress"
    for data in tqdm(dataloader, desc="desc"):
        user_id, history_items, history_items_len, \
            target_item_id, neg_item_id, user_features, item_features, neg_item_features, \
            item_pos_feedback, item_pos_feedback_len, neg_item_pos_feedbacks, neg_item_pos_feedbacks = data

        user_id = torch.tile(user_id, (num_test_neg_item+1, 1))
        item_idx = torch.cat([target_item_id, neg_item_id], dim=1)
        item_idx = torch.reshape(item_idx, (num_test_neg_item+1, 1))
        history_items = torch.tile(history_items, (num_test_neg_item+1, 1))
        history_items_len = torch.tile(history_items_len, (num_test_neg_item+1, 1))
        user_features = torch.tile(user_features, (num_test_neg_item+1, 1))
        item_features = item_features.unsqueeze(1)
        item_features = torch.cat([item_features, neg_item_features], dim=1)
        item_features = torch.reshape(item_features, (num_test_neg_item+1, -1))
        item_pos_feedback = torch.cat([item_pos_feedback.unsqueeze(1), neg_item_pos_feedbacks], dim=1)
        item_pos_feedback = torch.reshape(item_pos_feedback, (num_test_neg_item+1, -1))
        item_pos_feedback_len = torch.cat([item_pos_feedback_len.unsqueeze(1), neg_item_pos_feedbacks], dim=1)
        item_pos_feedback_len = torch.reshape(item_pos_feedback_len, (num_test_neg_item+1, -1))
        predictions = model.predict(user_id, item_idx, history_items, history_items_len, user_features, item_features,
                     item_pos_feedback, item_pos_feedback_len).squeeze(1)

        rank = predictions.argsort(descending=True).argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user