import pandas as pd
import pickle

# 读取数据
pd.set_option('display.max_columns', None)
data = pd.read_csv('./u.data', names=['user_id', 'item_id', 'rating', 'ts'], sep='\t', encoding='gbk')
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
user_num = data['user_id'].value_counts().size
item_num = data['item_id'].value_counts().size
print('user_num:', data['user_id'].value_counts().size)
print('item_num:', data['item_id'].value_counts().size)
meta_data = {'dataset_name': 'MovieLens100k', 'user_num': data['user_id'].value_counts().size, 'item_num': data['item_id'].value_counts().size}
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
# print(data)

# 处理标签和冷门item
data['label'] = (data['rating'] > threshold).astype(int)
#print(data['label'])
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
tmp_data = data[data['label'] == 1].sort_values(by=['user_id', 'ts'], ascending=[True, True])
val_data = tmp_data.groupby('user_id').apply(lambda x: x.iloc[-2])
test_data = tmp_data.groupby('user_id').apply(lambda x: x.iloc[-1])
merged_data = pd.concat([data, val_data, test_data])
train_data = merged_data.drop_duplicates(keep=False)
train_data = train_data.sort_values(by=['user_id', 'ts'], ascending=[True, True])
train_data = train_data[train_data['positive_behavior_offset'] >= 3]

train_data = train_data.drop(columns=['ts'])
val_data = val_data.drop(columns=['ts'])
test_data = test_data.drop(columns=['ts'])

# 存储数据
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 保存每个user交互过的positive item和negative item
data_positive = data_positive.sort_values(by=['user_id', 'ts'], ascending=[True, True])
user_item_interactions = data_positive.groupby('user_id')['item_id'].apply(list).to_dict()
for i in range(user_num):
    if i not in user_item_interactions:
        user_item_interactions[i] = []
pickle.dump(user_item_interactions, open('user_history_positive.pkl', 'wb'))
data_negative = data_negative.sort_values(by=['user_id', 'ts'], ascending=[True, True])
user_item_interactions = data_negative.groupby('user_id')['item_id'].apply(list).to_dict()
for i in range(user_num):
    if i not in user_item_interactions:
        user_item_interactions[i] = []
pickle.dump(user_item_interactions, open('user_history_negative.pkl', 'wb'))

# 保存每个item交互过的positive user和negative user
data_positive = data_positive.sort_values(by=['item_id', 'ts'], ascending=[True, True])
user_item_interactions = data_positive.groupby('item_id')['user_id'].apply(list).to_dict()
for i in range(item_num):
    if i not in user_item_interactions:
        user_item_interactions[i] = []
pickle.dump(user_item_interactions, open('item_history_positive.pkl', 'wb'))
data_negative = data_negative.sort_values(by=['item_id', 'ts'], ascending=[True, True])
user_item_interactions = data_negative.groupby('item_id')['user_id'].apply(list).to_dict()
for i in range(item_num):
    if i not in user_item_interactions:
        user_item_interactions[i] = []
pickle.dump(user_item_interactions, open('item_history_negative.pkl', 'wb'))

