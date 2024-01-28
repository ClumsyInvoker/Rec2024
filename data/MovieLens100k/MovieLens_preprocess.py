import pandas as pd
import pickle

# 读取数据
data = pd.read_csv('./u.data', names=['user_id', 'item_id', 'rating', 'ts'], sep='\t', encoding='gbk')
data['user_id'] = data['user_id'] - 1  # user id从0开始编号
data['item_id'] = data['item_id'] - 1  # item id从0开始编号

# 只选择rating大于阈值的数据
threshold = 1
data = data[data['rating'] >= threshold]

data = data.sort_values(by=['user_id', 'ts'], ascending=[True, True])

print('user_num:', data['user_id'].value_counts().size)
print('item_num:', data['item_id'].value_counts().size)
# 存储meta data
meta_data = {'dataset_name': 'MovieLens100k', 'user_num': data['user_id'].value_counts().size, 'item_num': data['item_id'].value_counts().size}
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

