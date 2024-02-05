import pandas as pd
import numpy as np
import pickle

user_num = 6040
item_num = 3706

data = pd.read_csv('./ratings.dat', names=['user_id', 'item_id', 'rating', 'ts'], sep='::', encoding='gbk', engine='python')
data['user_id'] = data['user_id'] - 1  # user id从0开始编号
data['item_id'] = data['item_id'] - 1  # item id从0开始编号

# 只选择rating大于阈值的数据
threshold = 3
data_positive = data[data['rating'] > threshold]
data_positive = data_positive.sort_values(['user_id', 'ts'], ascending=[True, True])
data_positive['positive_behavior_offset'] = data_positive.groupby('user_id').cumcount()
data_positive['positive_behavior_offset'] = data_positive.groupby(['user_id', 'ts'])['positive_behavior_offset'].transform('min')
data_positive = data_positive.sort_values(['item_id', 'ts'], ascending=[True, True])
data_positive['item_positive_behavior_offset'] = data_positive.groupby('item_id').cumcount()
data_positive['item_positive_behavior_offset'] = data_positive.groupby(['item_id', 'ts'])['item_positive_behavior_offset'].transform('min')

data_positive = data_positive.sort_values(by=['user_id', 'ts'], ascending=[True, True])
user_history_positive = data_positive.groupby('user_id')['item_id'].apply(list).to_dict()
user_history_positive_ts = data_positive.groupby('user_id')['ts'].apply(list).to_dict()
for i in range(user_num):
    if i not in user_history_positive:
        user_history_positive[i] = []
        user_history_positive_ts[i] = []

train_data = pd.read_csv("./train_data.csv")
cold_pos_item_data = train_data[(train_data['cold_item'] == 1) & (train_data['label'] == 1)]
random_samples = cold_pos_item_data.sample(n=500, random_state=42)

# 为冷启动正样本生成100个负样本
np.random.seed(0) # 固定随机种子
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def find_largest_index_less_than_target(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

user_history_positive = pickle.load(open('user_history_positive.pkl', 'rb'))
item_history_positive = pickle.load(open('item_history_positive.pkl', 'rb'))
user_features = pd.read_csv('user_features.csv')
item_features = pd.read_csv('item_features.csv')
neg_num = 100
max_length = 50
data_cold_item_test = []
for idx in range(len(random_samples)):
    row = random_samples.iloc[idx]
    user_id = int(row['user_id'])
    target_item_id = int(row['item_id'])
    positive_behavior_offset = int(row['positive_behavior_offset'])
    label = row['label']
    cold_item = row['cold_item']

    ts = row['ts']

    raw_history_items = user_history_positive[user_id][
                        max(0, positive_behavior_offset + 1 - max_length):positive_behavior_offset + 1]
    user_feature = user_features.iloc[user_id]
    item_feature = item_features.iloc[target_item_id]

    neg_user_ids = []
    neg_user_features = []
    neg_raw_history_items = []
    total_history_users = item_history_positive[target_item_id]
    for _ in range(neg_num):
        neg_user_id = random_neq(0, item_num, set(total_history_users + neg_user_ids))
        neg_user_ids.append(neg_user_id)
        neg_user_features.append(user_features.iloc[neg_user_id].values)

        p = find_largest_index_less_than_target(user_history_positive_ts[neg_user_id], ts)
        neg_raw_history_item = user_history_positive[neg_user_id][p + 1 - max_length:p + 1]
        neg_raw_history_items.append(neg_raw_history_item)

    data_cold_item_test.append({'user_id': user_id,
                                'raw_history_items': raw_history_items,
                                'target_item_id': target_item_id,
                                'user_feature': user_feature.values,
                                'item_feature': item_feature.values,
                                'neg_user_ids': neg_user_ids,
                                'neg_user_features': neg_user_features,
                                'neg_raw_history_items': neg_raw_history_items
                                })
pickle.dump(data_cold_item_test, open('data_cold_item_test.pkl', 'wb'))
# random_samples.to_csv('sample_cold_item_data.csv', index=False)
# pickle.dump(data_neg_items, open('sample_cold_item_100_neg.pkl', 'wb'))


