import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class DCN(nn.Module):
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
        if self.nume_feature_size != 0:
            self.fm_2nd_order_dense = nn.Linear(self.nume_feature_size, embed_dim)
        item_cate_voc_sizes = [voc_size for idx, voc_size in self.item_cate_id_feature_idx]  # 类别特征的词典大小
        item_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.item_cate_one_hot_feature_idx]  # 预留一个缺省embedding

        user_cate_voc_sizes = [voc_size for idx, voc_size in self.user_cate_id_feature_idx]
        user_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.user_cate_one_hot_feature_idx]
        self.item_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in item_cate_voc_sizes])  # 类别特征的二阶表示
        self.user_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in user_cate_voc_sizes])

        # item id / user id
        self.item_embedding = Embedding(num_embeddings=dim_config['item_id'], embed_dim=embed_dim)
        self.user_embedding = Embedding(num_embeddings=dim_config['user_id'], embed_dim=embed_dim)

        # Cross部分
        self.cross = CrossNetwork(input_dim=embed_dim, num_layers=2)

        # DNN 部分
        all_size = self.cate_feature_size+2+1 if self.nume_feature_size != 0 else self.cate_feature_size+2
        self.dnn = FullyConnectedLayer(input_size=all_size * embed_dim,
                                       hidden_size=hidden_dim + [embed_dim],
                                       bias=[True, True, False],
                                       batch_norm=True,
                                       dropout_rate=0.5,
                                       activation='relu',
                                       sigmoid=False
                                       )

        self.output_fc = nn.Linear(2*embed_dim, 1)

        # self.sigmoid = nn.Sigmoid() # 使用BCEWithLogitsLoss时不需要sigmoid

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        feature_emb = []
        item_emb = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]
        feature_emb.append(item_emb)
        feature_emb.append(user_emb)

        """FM 二阶部分"""
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            feature_emb.append(self.item_fm_2nd_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            feature_emb.append(torch.mm(cate_feature, self.item_fm_2nd_order_sparse_emb[i].weight))
            i += 1


        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            feature_emb.append(self.user_fm_2nd_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            feature_emb.append(torch.mm(cate_feature, self.user_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        if self.nume_feature_size != 0:
            dense_feature = []
            for idx in self.item_nume_feature_idx:
                dense_feature.append(item_features[:, idx].unsqueeze(1))
            for idx in self.user_nume_feature_idx:
                dense_feature.append(user_features[:, idx].unsqueeze(1))
            dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
            feature_emb.append(self.fm_2nd_order_dense(dense_feature))

        feature_emb = torch.stack(feature_emb, dim=1)  # [bs, cate_feature_size+2, embed_dim]

        cross_output = self.cross(feature_emb)  # [bs, cate_feature_size+2, embed_dim]
        cross_output = torch.mean(cross_output, dim=1)  # [bs, embed_dim]

        """DNN部分"""
        dnn_out = torch.flatten(feature_emb, 1)  # [bs, (cate_feature_size+2) * embed_dim]

        dnn_out = self.dnn(dnn_out)  # [bs, embed_dim]

        out = torch.cat([cross_output, dnn_out], dim=1)
        out = self.output_fc(out)

        # out = self.sigmoid(out)
        return out

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features)
