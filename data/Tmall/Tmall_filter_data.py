import pandas as pd

cnt = 0

def filter_data():
    data = pd.read_csv('user_log_format1.csv')

    data = data[data['action_type'].isin([0, 2])]
    # print(data.shape)

    purchase_data = data[data['action_type'] == 2]
    # print(purchase_data.shape)
    purchase_data.to_csv("purchase_data.csv", index=False)

    def filter(data):
        # 计算用户的记录次数
        user_interaction_counts = data.groupby('user_id').size().reset_index(name='user_interaction_count')

        # 筛选出记录次数大于等于50次的用户
        filtered_users = user_interaction_counts[user_interaction_counts['user_interaction_count'] >= 10]

        # 打印筛选结果
        print(filtered_users.shape)

        data = data[data['user_id'].isin(filtered_users['user_id'])]
        print(data.shape)


        # 计算商品的记录次数
        item_interaction_counts = data.groupby('item_id').size().reset_index(name='item_interaction_count')
        # 筛选出记录次数大于等于50次的item
        filtered_items = item_interaction_counts[item_interaction_counts['item_interaction_count'] >= 10]

        # 打印筛选结果
        print(filtered_items.shape)

        data = data[data['item_id'].isin(filtered_items['item_id'])]
        print(data.shape)

        return data

    for i in range(15):
        print(f'第{i+1}次筛选')
        purchase_data = filter(purchase_data)

    purchase_data.to_csv("filtered_purchase_data.csv", index=False)

    filterde_user_ids = purchase_data['user_id'].unique()
    filterde_item_ids = purchase_data['item_id'].unique()

    data = data[data['user_id'].isin(filterde_user_ids)]
    data = data[data['item_id'].isin(filterde_item_ids)]

    print(data.shape)
    data.to_csv("filtered_full_data.csv", index=False)

    data = data[['user_id','item_id','time_stamp','action_type']]
    data_positive = data[data['action_type'] == 2]
    user_history_positive = data_positive.groupby('user_id')['item_id'].apply(list).to_dict()

    # 处理action type同时为0和2的(u,i)对
    def process_group(group):
        global cnt
        cnt += 1
        if cnt % 2000 == 0:
            print(cnt)
        user_id = group['user_id'].values[0]
        group['item_id'] = group['item_id'].apply(lambda x: x if x not in user_history_positive[user_id] else -1)
        return group

    data = data.groupby(['user_id'], as_index=False).apply(process_group).reset_index(drop=True)

    data = data[data['item_id'] != -1]
    data = pd.concat([data, data_positive], axis=0)
    print(data.shape)
    data.to_csv("filtered_data.csv", index=False)


def generate_item_features():
    data = pd.read_csv('filtered_full_data.csv')
    grouped = data.groupby('item_id')
    item_ids = []
    cat_ids = []
    seller_ids = []
    brand_ids = []
    cnt = 0
    for name, group in grouped:
        item_ids.append(name)
        cat_ids.append(int(group['cat_id'].values[0]) if not pd.isna(group['cat_id'].values[0]) else -1)
        seller_ids.append(int(group['seller_id'].values[0]) if not pd.isna(group['seller_id'].values[0]) else -1)
        brand_ids.append(int(group['brand_id'].values[0]) if not pd.isna(group['brand_id'].values[0]) else -1)
        # if len(group['brand_id'].unique()) > 1:
        #     print(name)
        #     cnt += 1

    # print('multi_brand_id:', cnt)

    item_features = pd.DataFrame({'item_id': item_ids, 'cat_id': cat_ids, 'seller_id': seller_ids, 'brand_id': brand_ids})
    print(item_features)
    item_features.to_csv('raw_item_features.csv', index=False)

filter_data()
generate_item_features()