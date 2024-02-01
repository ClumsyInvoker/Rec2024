import json

meta_data = {}

MovieLens100k_mata_data = {"user_num": 943,
                           "item_num": 1682,
                           "user_feature": {
                               "dim": 23,
                               "nume_feat_idx": [0], # 0 age:0~1
                               "cate_id_feat_idx": [(1,2)], # 1 gender:0-1
                               "cate_one_hot_feat_idx": [(2, 21)] # 2-21 occupation: one hot
                               # 类别特征：id类是(idx, cate_num)，one hot类是(start_idx, end_idx)
                            },
                            "item_feature":{
                                "dim": 18,
                                "nume_feat_idx": [],
                                "cate_id_feat_idx": [],
                                "cate_one_hot_feat_idx": [(0, 17)] # 0-17 genre: one hot / multi hot
                            }
                           }
meta_data['MovieLens100k'] = MovieLens100k_mata_data

MovieLens1m_mata_data = {"user_num": 6040,
                         "item_num": 3706,
                         "user_feature": {
                               "dim": 23,
                               "nume_feat_idx": [0], # 0 age:0~1
                               "cate_id_feat_idx": [(1, 2)], # 1 gender:0-1
                               "cate_one_hot_feat_idx": [(2, 21)] # 2-21 occupation: one hot / multi hot
                               # 类别特征：id类是(idx, cate_num)，one hot类是(start_idx, end_idx)
                         },
                         "item_feature":{
                            "dim": 18,
                            "nume_feat_idx": [],
                            "cate_id_feat_idx": [],
                            "cate_one_hot_feat_idx": [(0, 17)] # 0-17 genre: one hot
                         }
                        }
meta_data['MovieLens1m'] = MovieLens1m_mata_data

json.dump(meta_data, open('dataset_meta_data.json', 'w'), indent=4)

dataset_meta_data = json.load(open('dataset_meta_data.json', 'r'))
print(dataset_meta_data)
