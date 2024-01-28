import pandas as pd
import pickle

# 读取数据
data = pd.read_csv('./ratings.dat', names=['user_id', 'item_id', 'rating', 'ts'], sep='::', encoding='gbk', engine='python')
data['user_id'] = data['user_id'] - 1  # user id从0开始编号
data['item_id'] = data['item_id'] - 1  # item id从0开始编号

# 只选择rating大于阈值的数据
threshold = 1
data = data[data['rating'] >= threshold]

data = data.sort_values(by=['user_id', 'ts'], ascending=[True, True])

# print('user_num:', data['user_id'].value_counts().size)
# print('item_num:', data['item_id'].value_counts().size)
print('user_num:', data['user_id'].max())
print('item_num:', data['item_id'].max())
# 存储meta data
meta_data = {'dataset_name': 'MovieLens1m', 'user_num': data['user_id'].max()+1, 'item_num': data['item_id'].max()+1}
meta_data = pd.DataFrame(meta_data, index=[0])
meta_data.to_csv('meta_data.csv', index=False)

# 对每个user取最后N个item
# grouped = data.groupby('user_id')
# N = 50  # 想取的行数
# data = grouped.tail(N)

selected_columns = data.iloc[:, [0,1,3]]
selected_columns.to_csv('data.txt', sep='\t', index=False, header=False)
selected_columns.to_csv('data.csv', index=False)

# 保存每个user交互过的item
user_item_interactions = data.groupby('user_id')['item_id'].apply(list).to_dict()
pickle.dump(user_item_interactions, open('user_history.pkl', 'wb'))

# 保存每个item交互过的user
data = data.sort_values(by=['item_id', 'ts'], ascending=[True, True])
user_item_interactions = data.groupby('item_id')['user_id'].apply(list).to_dict()
pickle.dump(user_item_interactions, open('item_history.pkl', 'wb'))

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
item_features = item_features.append(pd.DataFrame(missing_item), ignore_index=True)
item_features = item_features.sort_values(by=['item_id'])
item_features = item_features.drop(columns=['item_id', 'title', 'genres'])
item_features.to_csv('item_features.csv', index=False)



