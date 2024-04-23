import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer


class CDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        hidden_dim = [256, 128]
        self.enable_behavior_embedding = config[
            'enable_behavior_embedding'] if 'enable_behavior_embedding' in config else False
        self.other_features_size = 3 if self.enable_behavior_embedding else 2  # user behavior, user id, item id

        self.item_nume_feature_idx = config['item_feature']['nume_feat_idx']  # 数值特征
        self.item_cate_id_feature_idx = config['item_feature']['cate_id_feat_idx']  # 类别id特征
        self.item_cate_one_hot_feature_idx = config['item_feature']['cate_one_hot_feat_idx']  # 类别one-hot特征
        self.user_nume_feature_idx = config['user_feature']['nume_feat_idx']
        self.user_cate_id_feature_idx = config['user_feature']['cate_id_feat_idx']
        self.user_cate_one_hot_feature_idx = config['user_feature']['cate_one_hot_feat_idx']

        self.nume_feature_size = len(self.item_nume_feature_idx) + len(self.user_nume_feature_idx)  # 数值特征的个数
        self.user_nume_feature_size = len(self.user_nume_feature_idx)
        self.item_nume_feature_size = len(self.item_nume_feature_idx)
        self.cate_feature_size = len(self.item_cate_id_feature_idx) + len(self.item_cate_one_hot_feature_idx) + \
                                 len(self.user_cate_id_feature_idx) + len(self.user_cate_one_hot_feature_idx)  # 类别特征的个数

        # embedding layers
        if self.user_nume_feature_size != 0:
            self.user_fm_2nd_order_dense = nn.Linear(self.user_nume_feature_size, embed_dim)
        if self.item_nume_feature_size != 0:
            self.item_fm_2nd_order_dense = nn.Linear(self.item_nume_feature_size, embed_dim)
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

        # User 部分
        self.user_cate_feature_size = len(self.user_cate_id_feature_idx) + len(self.user_cate_one_hot_feature_idx)
        self.user_other_features_size = 2 if self.enable_behavior_embedding else 1  # user behavior, user id
        user_all_size = self.user_cate_feature_size + self.user_other_features_size + 1 if self.user_nume_feature_size != 0 \
            else self.user_cate_feature_size + self.user_other_features_size
        self.user_dnn = FullyConnectedLayer(input_size=user_all_size * embed_dim,
                                       hidden_size=hidden_dim + [embed_dim],
                                       bias=[True, True, False],
                                       batch_norm=True,
                                       dropout_rate=0.5,
                                       activation='relu',
                                       sigmoid=False
                                       )
        self.user_dnn_r = FullyConnectedLayer(input_size=embed_dim,
                                       hidden_size=[embed_dim],
                                       bias=[False],
                                       batch_norm=True,
                                       dropout_rate=0.5,
                                       activation='relu',
                                       sigmoid=False
                                       )
        self.user_dnn_m = FullyConnectedLayer(input_size=embed_dim,
                                              hidden_size=[embed_dim],
                                              bias=[False],
                                              batch_norm=True,
                                              dropout_rate=0.5,
                                              activation='relu',
                                              sigmoid=False
                                              )

        # Item 部分
        self.item_cate_feature_size = len(self.item_cate_id_feature_idx) + len(self.item_cate_one_hot_feature_idx)
        item_all_size = self.item_cate_feature_size + 2 if self.item_nume_feature_size != 0 \
            else self.item_cate_feature_size + 1
        self.item_gate = Embedding(num_embeddings=2, embed_dim=item_all_size)

        self.output_fc = nn.Linear(2*embed_dim, 1)

        # self.sigmoid = nn.Sigmoid() # 使用BCEWithLogitsLoss时不需要sigmoid

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, cold_item, mode='m'):
        item_feature_emb = []
        user_feature_emb = []
        item_emb = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]
        item_feature_emb.append(item_emb)
        user_feature_emb.append(user_emb)

        if self.enable_behavior_embedding:
            # user behavior
            user_behavior_embedded = self.item_embedding(history_item_id) # (batch_size, max_history_len, embed_dim)
            mask = history_len.bool().unsqueeze(-1) # (batch_size, max_history_len, 1)
            mask = torch.tile(mask, [1, 1, user_behavior_embedded.shape[-1]]) # (batch_size, max_history_len, embed_dim)
            user_behavior_embedded = torch.where(mask, user_behavior_embedded, torch.zeros_like(user_behavior_embedded))
            user_behavior_embedded = torch.sum(user_behavior_embedded, dim=1) / torch.sum(history_len, dim=1, keepdim=True) # (batch_size, embed_dim)
            user_behavior_embedded = torch.where(torch.isnan(user_behavior_embedded), torch.zeros_like(user_behavior_embedded), user_behavior_embedded)

            user_feature_emb.append(user_behavior_embedded)

        """FM 二阶部分"""
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            item_feature_emb.append(self.item_fm_2nd_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            item_feature_emb.append(torch.mm(cate_feature, self.item_fm_2nd_order_sparse_emb[i].weight))
            i += 1


        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            user_feature_emb.append(self.user_fm_2nd_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            user_feature_emb.append(torch.mm(cate_feature, self.user_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        if self.item_nume_feature_size != 0:
            item_dense_feature = []
            for idx in self.item_nume_feature_idx:
                item_dense_feature.append(item_features[:, idx].unsqueeze(1))
            item_dense_feature = torch.cat(item_dense_feature, dim=1)  # [bs, nume_feature_size]
            item_feature_emb.append(self.item_fm_2nd_order_dense(item_feature_emb))

        if self.user_nume_feature_size != 0:
            user_dense_feature = []
            for idx in self.user_nume_feature_idx:
                user_dense_feature.append(user_features[:, idx].unsqueeze(1))
            user_dense_feature = torch.cat(user_dense_feature, dim=1)  # [bs, nume_feature_size]
            user_feature_emb.append(self.user_fm_2nd_order_dense(user_dense_feature))

        # user部分
        user_feature_emb = torch.stack(user_feature_emb, dim=1)  # [bs, cate_feature_size+2, embed_dim]
        dnn_out = torch.flatten(user_feature_emb, 1)  # [bs, (cate_feature_size+2) * embed_dim]

        dnn_out = self.user_dnn(dnn_out)  # [bs, embed_dim]

        if mode == 'r':
            user_final_embedding = self.user_dnn_r(dnn_out)
        else:
            user_final_embedding = self.user_dnn_m(dnn_out)

        # item部分
        item_gate = self.item_gate(cold_item.type(torch.int).squeeze(-1))
        item_gate = F.softmax(item_gate, dim=1).unsqueeze(-1)
        item_feature_emb = torch.stack(item_feature_emb, dim=1)  # [bs, cate_feature_size+2, embed_dim]
        item_final_embedding = torch.sum(item_gate * item_feature_emb, dim = 1, keepdim=False)  # [bs, embed_dim]


        # 输出
        out = torch.cat([user_final_embedding, item_final_embedding], dim=1)
        out = self.output_fc(out)

        # out = self.sigmoid(out)
        return out

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, cold_item, mode='m'):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features, cold_item, mode='m')