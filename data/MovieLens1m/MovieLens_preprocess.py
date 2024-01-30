import pandas as pd
import pickle
import numpy as np

# 二分查找
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


# 读取数据
pd.set_option('display.max_columns', None)
data = pd.read_csv('./ratings.dat', names=['user_id', 'item_id', 'rating', 'ts'], sep='::', encoding='gbk', engine='python')
data['user_id'] = data['user_id'] - 1  # user id从0开始编号
data['item_id'] = data['item_id'] - 1  # item id从0开始编号

# 只选择rating大于阈值的数据
threshold = 3
data_positive = data[data['rating'] > threshold]
data_negative = data[data['rating'] <= threshold]

# 统计热门item和冷门item
item_count = data_positive['item_id'].value_counts()
item_count = pd.DataFrame({'item_id': item_count.index, 'count': item_count.values})
print("total interaction num: {}, positive interaction num: {}, negative interaction num: {}"
      .format(data.shape[0], data_positive.shape[0], data_negative.shape[0]))
print("total item num: {}, positive item num: {}"
      .format(data['item_id'].value_counts().shape[0], data_positive['item_id'].value_counts().shape[0]))
#统计交互次数小于10的item
item_count = item_count[item_count['count'] < 10]
print("item num with interaction less than 10: {}".format(item_count.shape[0]))
cold_item_ids = item_count['item_id'].values


# 存储meta data
user_num = data['user_id'].max()
item_num = data['item_id'].max()
# print('user_num:', data['user_id'].value_counts().size)
# print('item_num:', data['item_id'].value_counts().size)
print('user_num:', data['user_id'].max())
print('item_num:', data['item_id'].max())
meta_data = {'dataset_name': 'MovieLens1m', 'user_num': data['user_id'].max()+1, 'item_num': data['item_id'].max()+1}
meta_data = pd.DataFrame(meta_data, index=[0])
meta_data.to_csv('meta_data.csv', index=False)

# 预处理，计算每个用户和物品在每个时间点之前的正样本数量
data_positive = data_positive.sort_values(['user_id', 'ts'], ascending=[True, True])
data_positive['positive_behavior_offset'] = data_positive.groupby('user_id').cumcount()
data_positive['positive_behavior_offset'] = data_positive.groupby(['user_id', 'ts'])['positive_behavior_offset'].transform('min')

data_positive = data_positive.sort_values(['item_id', 'ts'], ascending=[True, True])
data_positive['item_positive_behavior_offset'] = data_positive.groupby('item_id').cumcount()
data_positive['item_positive_behavior_offset'] = data_positive.groupby(['item_id', 'ts'])['item_positive_behavior_offset'].transform('min')

data_negative = data_negative.sort_values(['item_id', 'ts'], ascending=[True, True])
data_negative['item_negative_behavior_offset'] = data_negative.groupby('item_id').cumcount()
data_negative['item_negative_behavior_offset'] = data_negative.groupby(['item_id', 'ts'])['item_negative_behavior_offset'].transform('min')

# 合并到原始数据集
data = data.merge(data_positive[['user_id', 'item_id', 'ts', 'positive_behavior_offset', 'item_positive_behavior_offset']], on=['user_id', 'item_id', 'ts'], how='left')
# 填充缺失值
data = data.sort_values(by=['user_id', 'ts'], ascending=[True, True])
data['positive_behavior_offset'] = data['positive_behavior_offset'].ffill()
data['positive_behavior_offset'] = data['positive_behavior_offset'].fillna(0)
data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].ffill()
data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].fillna(0)

data = data.merge(data_negative[['item_id', 'ts', 'item_negative_behavior_offset']], on=['item_id', 'ts'], how='left')
data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].ffill()
data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].fillna(0)
# data = data.drop_duplicates(keep=False)

# 处理标签和冷门item
data['label'] = (data['rating'] > threshold).astype(int)
data['cold_item'] = data['item_id'].isin(cold_item_ids).astype(int)

# 填充缺失值
data.fillna({'positive_behavior_offset': 0, 'item_positive_behavior_offset': 0, 'item_negative_behavior_offset': 0}, inplace=True)

# 转成int类型
data['user_id'] = data['user_id'].astype(int)
data['item_id'] = data['item_id'].astype(int)
data['positive_behavior_offset'] = data['positive_behavior_offset'].astype(int)
data['item_positive_behavior_offset'] = data['item_positive_behavior_offset'].astype(int)
data['item_negative_behavior_offset'] = data['item_negative_behavior_offset'].astype(int)

# 划分数据集
def get_second_to_last_row(group):
    if len(group) >= 2:
        return group.iloc[-2]
    else:
        # 如果行数不足2行，返回None或者适当的默认值
        return None

def get_last_row(group):
    if len(group) >= 1:
        return group.iloc[-1]
    else:
        # 如果行数不足2行，返回None或者适当的默认值
        return None

tmp_data = data[data['label'] == 1].sort_values(by=['user_id', 'ts'], ascending=[True, True])
val_data = tmp_data.groupby('user_id').apply(get_second_to_last_row)
val_data = val_data.dropna()
test_data = tmp_data.groupby('user_id').apply(get_last_row)
test_data = test_data.dropna()
merged_data = pd.concat([data, val_data, test_data])
train_data = merged_data.drop_duplicates(keep=False)
train_data = train_data.sort_values(by=['user_id', 'ts'], ascending=[True, True])
train_data = train_data[train_data['positive_behavior_offset'] >= 3]

# train_data = train_data.drop(columns=['ts'])
# val_data = val_data.drop(columns=['ts'])
# test_data = test_data.drop(columns=['ts'])

# 存储数据
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 保存每个user交互过的positive item和negative item
data_positive = data_positive.sort_values(by=['user_id', 'ts'], ascending=[True, True])
user_history_positive = data_positive.groupby('user_id')['item_id'].apply(list).to_dict()
user_history_positive_ts = data_positive.groupby('user_id')['ts'].apply(list).to_dict()
for i in range(user_num):
    if i not in user_history_positive:
        user_history_positive[i] = []
        user_history_positive_ts[i] = []
pickle.dump(user_history_positive, open('user_history_positive.pkl', 'wb'))

data_negative = data_negative.sort_values(by=['user_id', 'ts'], ascending=[True, True])
user_history_negative = data_negative.groupby('user_id')['item_id'].apply(list).to_dict()
user_history_negative_ts = data_negative.groupby('user_id')['ts'].apply(list).to_dict()
for i in range(user_num):
    if i not in user_history_negative:
        user_history_negative[i] = []
        user_history_negative_ts[i] = []
pickle.dump(user_history_negative, open('user_history_negative.pkl', 'wb'))

# 保存每个item交互过的positive user和negative user
data_positive = data_positive.sort_values(by=['item_id', 'ts'], ascending=[True, True])
item_history_positive = data_positive.groupby('item_id')['user_id'].apply(list).to_dict()
item_history_positive_ts = data_positive.groupby('item_id')['ts'].apply(list).to_dict()
for i in range(item_num):
    if i not in item_history_positive:
        item_history_positive[i] = []
        item_history_positive_ts[i] = []
pickle.dump(item_history_positive, open('item_history_positive.pkl', 'wb'))

data_negative = data_negative.sort_values(by=['item_id', 'ts'], ascending=[True, True])
item_history_negative = data_negative.groupby('item_id')['user_id'].apply(list).to_dict()
item_history_negative_ts = data_negative.groupby('item_id')['ts'].apply(list).to_dict()
for i in range(item_num):
    if i not in item_history_negative:
        item_history_negative[i] = []
        item_history_negative_ts[i] = []
pickle.dump(item_history_negative, open('item_history_negative.pkl', 'wb'))

# 保存item的特征
item_features = pd.read_csv('./movies.dat', names=['item_id', 'title', 'genres'], sep='::', encoding='ISO-8859-1', engine='python')
item_features['item_id'] = item_features['item_id'] - 1
genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
         'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical',
         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
for genre in genres:
    item_features[genre] = item_features['genres'].apply(lambda x: int(genre in x.split('|')))
#补充缺失的id
missing_item = {'item_id': [i for i in range(item_features['item_id'].max()+1) if i not in item_features['item_id'].values]}
for genre in genres:
    missing_item[genre] = [0]*len(missing_item['item_id'])
item_features = item_features._append(pd.DataFrame(missing_item), ignore_index=True)
item_features = item_features.sort_values(by=['item_id'])
item_features = item_features.drop(columns=['item_id', 'title', 'genres'])
item_features.to_csv('item_features.csv', index=False)


# 为验证集和测试集生成100个负样本
np.random.seed(0) # 固定随机种子
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

neg_num = 100
feedback_max_length = 10
data_neg_items = []
data_neg_item_pos_feedbacks = []
for idx in range(len(val_data)):
    user_id = int(val_data.iloc[idx]['user_id'])
    ts = int(val_data.iloc[idx]['ts'])
    neg_item_ids = []
    raw_neg_item_pos_feedbacks = []
    total_history_items = user_history_positive[user_id][:-1]
    for _ in range(neg_num):
        neg_item_id = random_neq(0, item_num, set(total_history_items + neg_item_ids))
        p = find_largest_index_less_than_target(item_history_positive_ts[neg_item_id], ts) # 防止特征穿越
        raw_neg_item_pos_feedback = item_history_positive[neg_item_id][p+1-feedback_max_length:p+1]

        neg_item_ids.append(neg_item_id)
        raw_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedback)
    data_neg_items.append(neg_item_ids)
    data_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedbacks)
print("neg_item_id 0: ", data_neg_items[0])
pickle.dump(data_neg_items, open('val_data_neg_items.pkl', 'wb'))
pickle.dump(data_neg_item_pos_feedbacks, open('val_data_neg_item_pos_feedbacks.pkl', 'wb'))

data_neg_items = []
data_neg_item_pos_feedbacks = []
for idx in range(len(test_data)):
    user_id = int(test_data.iloc[idx]['user_id'])
    ts = int(test_data.iloc[idx]['ts'])
    neg_item_ids = []
    raw_neg_item_pos_feedbacks = []
    total_history_items = user_history_positive[user_id]
    for _ in range(neg_num):
        neg_item_id = random_neq(0, item_num, set(total_history_items + neg_item_ids))
        p = find_largest_index_less_than_target(item_history_positive_ts[neg_item_id], ts)
        raw_neg_item_pos_feedback = item_history_positive[neg_item_id][p + 1 - feedback_max_length:p + 1]

        neg_item_ids.append(neg_item_id)
        raw_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedback)
    data_neg_items.append(neg_item_ids)
    data_neg_item_pos_feedbacks.append(raw_neg_item_pos_feedbacks)
pickle.dump(data_neg_items, open('test_data_neg_items.pkl', 'wb'))
pickle.dump(data_neg_item_pos_feedbacks, open('test_data_neg_item_pos_feedbacks.pkl', 'wb'))