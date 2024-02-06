from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader


def evaluate_by_model_name(model_name, model, dataset, args, mode='test'):
    if model_name == "DSSM_SASRec_PTCR" or "PTCR" in model_name:
        # 使用prompt的模型，需要正反馈信息
        return evaluate_prompt(model, dataset, args, mode)
    elif model_name == "PLATE" or model_name == "MetaEmb":
        # 需要cold item信息的模型
        return evaluate_PLATE(model, dataset, args, mode)
    else:
        return evaluate(model, dataset, args)


def evaluate(model, dataset, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0
    top_K = getattr(args, 'top_K', 10)

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
        predictions = model.predict(user_id, item_idx, history_items, history_items_len, user_features, item_features).squeeze(1)

        rank = predictions.argsort(descending=True).argsort()[0].item()

        valid_user += 1

        if rank < top_K:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

def evaluate_prompt(model, dataset, args, mode='test'):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0
    top_K = getattr(args, 'top_K', 10)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_test_neg_item = args.num_test_neg_item
    desc = "Testing Progress" if mode == 'test' else "Validating Progress"
    for data in tqdm(dataloader, desc=desc):
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

        if rank < top_K:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_PLATE(model, dataset, args, mode='test'):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0
    top_K = getattr(args, 'top_K', 10)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_test_neg_item = args.num_test_neg_item
    for data in tqdm(dataloader, desc="{} Progress".format(mode)):
        user_id, history_items, history_items_len, \
        target_item_id, neg_item_id, user_features, item_features, neg_item_features, \
            is_cold_item = data

        user_id = torch.tile(user_id, (num_test_neg_item + 1, 1))
        item_idx = torch.cat([target_item_id, neg_item_id], dim=1)
        item_idx = torch.reshape(item_idx, (num_test_neg_item + 1, 1))
        history_items = torch.tile(history_items, (num_test_neg_item + 1, 1))
        history_items_len = torch.tile(history_items_len, (num_test_neg_item + 1, 1))
        user_features = torch.tile(user_features, (num_test_neg_item + 1, 1))
        item_features = item_features.unsqueeze(1)
        item_features = torch.cat([item_features, neg_item_features], dim=1)
        item_features = torch.reshape(item_features, (num_test_neg_item + 1, -1))
        is_cold_item = torch.reshape(is_cold_item, (num_test_neg_item + 1, -1))
        predictions = model.predict(user_id, item_idx, history_items, history_items_len, user_features,
                                    item_features, is_cold_item).squeeze(1)

        rank = predictions.argsort(descending=True).argsort()[0].item()

        valid_user += 1

        if rank < top_K:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user