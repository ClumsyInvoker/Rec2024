import pandas as pd

def item_features_process():
    raw_item_features = pd.read_csv('raw_item_features.csv')
    # print(raw_item_features['cat_id'].value_counts())
    # print(raw_item_features['seller_id'].value_counts())
    # print(raw_item_features['brand_id'].value_counts())

    item_features = raw_item_features
    item_features['raw_item_id'] = item_features['item_id']

    # 处理原始的item id
    item_id_counts = item_features['item_id'].unique().tolist()
    item_id_counts = sorted(item_id_counts)
    item_features['item_id'] = item_features['item_id'].map(lambda x: item_id_counts.index(x))
    # print(item_features)

    # 处理cat_id
    cat_id_counts = raw_item_features['cat_id'].unique().tolist()
    cat_id_counts = sorted(cat_id_counts)
    item_features['cat_id'] = item_features['cat_id'].map(lambda x: cat_id_counts.index(x))
    # print(item_features)

    # 处理seller_id
    seller_id_counts = raw_item_features['seller_id'].unique().tolist()
    seller_id_counts = sorted(seller_id_counts)
    item_features['seller_id'] = item_features['seller_id'].map(lambda x: seller_id_counts.index(x))
    # print(item_features)

    # 处理brand_id
    brand_id_counts = raw_item_features['brand_id'].unique().tolist()
    brand_id_counts = sorted(brand_id_counts)
    item_features['brand_id'] = item_features['brand_id'].map(lambda x: brand_id_counts.index(x))
    # print(item_features)

    item_features.to_csv('reid_item_features.csv', index=False)


def user_features_process():
    raw_user_features = pd.read_csv('user_info_format1.csv')
    data = pd.read_csv('filtered_purchase_data.csv')
    filtered_user_ids = data['user_id'].unique().tolist()
    raw_user_features = raw_user_features[raw_user_features['user_id'].isin(filtered_user_ids)]
    # print(raw_user_features)

    user_features = raw_user_features
    user_features['raw_user_id'] = user_features['user_id']

    # 处理原始的user id
    user_id_counts = user_features['user_id'].unique().tolist()
    user_id_counts = sorted(user_id_counts)
    user_features['user_id'] = user_features['user_id'].map(lambda x: user_id_counts.index(x))
    # print(user_features)

    # 处理age_range，共10种取值
    # print(user_features['age_range'].value_counts())
    voc_size = user_features['age_range'].max() + 1
    print('age_range:', voc_size)
    user_features['age_range'] = user_features['age_range'].map(lambda x: int(x) if not pd.isna(x) else voc_size)
    # print(user_features)

    # 处理gender，共4种取值
    # print(user_features['gender'].value_counts())
    voc_size = user_features['gender'].max() + 1
    print('gender:', voc_size)
    user_features['gender'] = user_features['gender'].map(lambda x: int(x) if not pd.isna(x) else voc_size) # 处理缺失值
    # print(user_features)
    # print(user_features['gender'].value_counts())

    user_features.to_csv('reid_user_features.csv', index=False)

item_features_process()
user_features_process()
