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

KuaiRand_mata_data = {"user_num": 27284,
                         "item_num": 7582,
                         "user_feature": {
                               "dim": 25,
                               "nume_feat_idx": [],
                               "cate_id_feat_idx": [(0, 4), (1, 2), (2, 2), (3, 8), (4, 7),
                                                    (5, 7), (6, 7), (7, 2), (8, 7), (9, 50),
                                                    (10, 1471), (11, 15), (12, 34), (13, 3),
                                                    (14, 118), (15, 454), (16, 7), (17, 5), (18, 5), (19, 2),
                                                    (20, 2), (21, 2), (22, 2), (23, 2), (24, 2)], #
                               "cate_one_hot_feat_idx": []
                               # 类别特征：id类是(idx, cate_num)，one hot类是(start_idx, end_idx)
                         },
                         "item_feature":{
                            "dim": 55,
                            "nume_feat_idx": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                              34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                                              44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                                              54],
                            "cate_id_feat_idx": [(0,2), (1,4), (2,6), (3,69)],
                            "cate_one_hot_feat_idx": []
                         }
                        }
meta_data['KuaiRand'] = KuaiRand_mata_data

json.dump(meta_data, open('dataset_meta_data.json', 'w'), indent=4)

dataset_meta_data = json.load(open('dataset_meta_data.json', 'r'))
print(dataset_meta_data)
