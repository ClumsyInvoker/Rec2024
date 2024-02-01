import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer


class DeepFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        hidden_dim = [256, 128]

        self.item_nume_feature_idx = config['item_feature']['nume_feat_idx']  # 数值特征
        self.item_cate_id_feature_idx = config['item_feature']['cate_id_feat_idx']  # 类别id特征
        self.item_cate_one_hot_feature_idx = config['item_feature']['cate_one_hot_feat_idx']  # 类别one-hot特征
        self.user_nume_feature_idx = config['user_feature']['nume_feat_idx']
        self.user_cate_id_feature_idx = config['user_feature']['cate_id_feat_idx']
        self.user_cate_one_hot_feature_idx = config['user_feature']['cate_one_hot_feat_idx']

        self.nume_feature_size = len(self.item_nume_feature_idx) + len(self.user_nume_feature_idx)  # 数值特征的个数
        self.cate_feature_size = len(self.item_cate_id_feature_idx) + len(self.item_cate_one_hot_feature_idx) + \
                                 len(self.user_cate_id_feature_idx) + len(self.user_cate_one_hot_feature_idx)  # 类别特征的个数

        # embedding layers
        """FM部分"""
        # 一阶
        if self.nume_feature_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_feature_size, 1)  # 数值特征的一阶表示
        item_cate_voc_sizes = [voc_size for idx, voc_size in self.item_cate_id_feature_idx]  # 类别特征的词典大小
        item_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.item_cate_one_hot_feature_idx]  # 预留一个缺省embedding
        self.item_fm_1st_order_sparse_emb = nn.ModuleList([
            Embedding(voc_size, 1) for voc_size in item_cate_voc_sizes])  # 类别特征的一阶表示
        user_cate_voc_sizes = [voc_size for idx, voc_size in self.user_cate_id_feature_idx]
        user_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.user_cate_one_hot_feature_idx]
        self.user_fm_1st_order_sparse_emb = nn.ModuleList([
            Embedding(voc_size, 1) for voc_size in user_cate_voc_sizes])

        # 二阶
        self.item_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in item_cate_voc_sizes])  # 类别特征的二阶表示
        self.user_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in user_cate_voc_sizes])
        # item id / user id
        self.item_embedding = Embedding(num_embeddings=dim_config['item_id'], embed_dim=embed_dim)
        self.user_embedding = Embedding(num_embeddings=dim_config['user_id'], embed_dim=embed_dim)

        """DNN部分"""
        self.dense_linear = nn.Linear(self.nume_feature_size, (self.cate_feature_size+2) * embed_dim)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()

        # for DNN
        self.dnn = FullyConnectedLayer(input_size=(self.cate_feature_size+2) * embed_dim,
                                       hidden_size=hidden_dim + [1],
                                       bias=[True, True, False],
                                       batch_norm=True,
                                       dropout_rate=0.5,
                                       activation='relu',
                                       sigmoid=False
                                       )

        # self.sigmoid = nn.Sigmoid() # 使用BCEWithLogitsLoss时不需要sigmoid

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        """FM 一阶部分"""
        item_emb = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]

        fm_1st_sparse_res = []
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            fm_1st_sparse_res.append(self.item_fm_1st_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float() # [bs, end_idx - start_idx + 1 + 1]
            fm_1st_sparse_res.append(torch.mm(cate_feature, self.item_fm_1st_order_sparse_emb[i].weight))
            i += 1

        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            fm_1st_sparse_res.append(self.user_fm_1st_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_1st_sparse_res.append(torch.mm(cate_feature, self.user_fm_1st_order_sparse_emb[i].weight))
            i += 1

        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_feature_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]


        if self.nume_feature_size != 0:
            dense_feature = []
            for idx in self.item_nume_feature_idx:
                dense_feature.append(item_features[:, idx].unsqueeze(1))
            for idx in self.user_nume_feature_idx:
                dense_feature.append(user_features[:, idx].unsqueeze(1))
            dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
            fm_1st_dense_res = self.fm_1st_order_dense(dense_feature)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_sparse_res = []
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            fm_2nd_sparse_res.append(self.item_fm_2nd_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_2nd_sparse_res.append(torch.mm(cate_feature, self.item_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            fm_2nd_sparse_res.append(self.user_fm_2nd_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_2nd_sparse_res.append(torch.mm(cate_feature, self.user_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        fm_2nd_sparse_res.append(item_emb)
        fm_2nd_sparse_res.append(user_emb)
        fm_2nd_concat_1d = torch.stack(fm_2nd_sparse_res, dim=1)  # [bs, cate_feature_size+2, embed_dim]

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, embed_dim]
        square_sum_embed = sum_embed * sum_embed  # [bs, embed_dim]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, cate_feature_size+2, embed_dim]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, embed_dim]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, embed_dim]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, (cate_feature_size+2) * embed_dim]

        if self.nume_feature_size != 0:
            dense_feature = []
            for idx in self.item_nume_feature_idx:
                dense_feature.append(item_features[:, idx].unsqueeze(1))
            for idx in self.user_nume_feature_idx:
                dense_feature.append(user_features[:, idx].unsqueeze(1))
            dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
            dense_out = self.relu(self.dense_linear(dense_feature))  # [bs, (cate_feature_size+2) * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, (cate_feature_size+2) * emb_size]

        dnn_out = self.dnn(dnn_out)  # [bs, 1]

        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
        # out = self.sigmoid(out)
        return out

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features)
