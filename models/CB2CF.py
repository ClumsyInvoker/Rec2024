import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer
from models.SASRec import SASRec


class CB2CF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']

        self.item_nume_feature_idx = config['item_feature']['nume_feat_idx']  # 数值特征
        self.item_cate_id_feature_idx = config['item_feature']['cate_id_feat_idx']  # 类别id特征
        self.item_cate_one_hot_feature_idx = config['item_feature']['cate_one_hot_feat_idx']  # 类别one-hot特征
        self.user_nume_feature_idx = config['user_feature']['nume_feat_idx']
        self.user_cate_id_feature_idx = config['user_feature']['cate_id_feat_idx']
        self.user_cate_one_hot_feature_idx = config['user_feature']['cate_one_hot_feat_idx']

        self.nume_feature_size = len(self.item_nume_feature_idx) + len(self.user_nume_feature_idx)  # 数值特征的个数
        self.cate_feature_size = len(self.item_cate_id_feature_idx) + len(self.item_cate_one_hot_feature_idx) + \
                                 len(self.user_cate_id_feature_idx) + len(self.user_cate_one_hot_feature_idx)  # 类别特征的个数
        self.user_nume_feature_size = len(self.user_nume_feature_idx)
        self.user_cate_feature_size = len(self.user_cate_id_feature_idx) + len(self.user_cate_one_hot_feature_idx)
        self.user_total_feature_size = self.user_cate_feature_size + 1 if self.user_nume_feature_size != 0 else self.cate_feature_size

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

        # feature dense
        self.item_feature_dense = FullyConnectedLayer(input_size=dim_config['item_feature'],
                                                      hidden_size=[embed_dim],
                                                      bias=[True],
                                                      activation='sigmoid')
        self.user_feature_dense = FullyConnectedLayer(input_size=dim_config['user_feature'],
                                                      hidden_size=[embed_dim],
                                                      bias=[True],
                                                      activation='sigmoid')

        self.user_fusion_layer = nn.Linear(self.user_total_feature_size*embed_dim, embed_dim)

        self.user_sequential_model = SASRec(config)

        self.item_fc_layer = FullyConnectedLayer(input_size=2*embed_dim,
                                            hidden_size=[embed_dim],
                                            bias=[False],
                                            activation='relu',
                                            sigmoid=False) # 如果是BCEWithLogitsLoss， 最后一层不需要sigmoid

        # self.sigmoid = nn.Sigmoid() # 使用BCEWithLogitsLoss时不需要sigmoid

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        target_item_embedded = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]

        # feature_emb = []
        # """FM 二阶部分"""
        # # item类别特征处理
        # i = 0
        # for idx, voc_size in self.item_cate_id_feature_idx:
        #     feature_emb.append(self.item_fm_2nd_order_sparse_emb[i](item_features[:, idx].long()))
        #     i += 1
        # for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
        #     cate_feature = item_features[:, start_idx: end_idx+1].long()
        #     if len(cate_feature.shape) == 1:
        #         cate_feature = cate_feature.unsqueeze(1)
        #     padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
        #     cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
        #     feature_emb.append(torch.mm(cate_feature, self.item_fm_2nd_order_sparse_emb[i].weight))
        #     i += 1
        #
        # # user类别特征处理
        # i = 0
        # for idx, voc_size in self.user_cate_id_feature_idx:
        #     feature_emb.append(self.user_fm_2nd_order_sparse_emb[i](user_features[:, idx].long()))
        #     i += 1
        # for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
        #     cate_feature = user_features[:, start_idx: end_idx+1].long()
        #     if len(cate_feature.shape) == 1:
        #         cate_feature = cate_feature.unsqueeze(1)
        #     padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
        #     cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
        #     feature_emb.append(torch.mm(cate_feature, self.user_fm_2nd_order_sparse_emb[i].weight))
        #     i += 1
        #
        # if self.nume_feature_size != 0:
        #     dense_feature = []
        #     for idx in self.item_nume_feature_idx:
        #         dense_feature.append(item_features[:, idx].unsqueeze(1))
        #     for idx in self.user_nume_feature_idx:
        #         dense_feature.append(user_features[:, idx].unsqueeze(1))
        #     dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
        #     feature_emb.append(self.fm_2nd_order_dense(dense_feature))
        #
        # feature_emb = torch.conat(feature_emb, dim=1)  # [bs, total_feature_size * embed_dim]
        # feature_emb = self.fusion_layer(feature_emb)

        item_feature_embedded = self.item_feature_dense(item_features)  # (batch_size, embed_dim)
        user_feature_embedded = self.user_feature_dense(user_features)  # (batch_size, embed_dim)

        """behavior部分"""
        user_behavior_embedded = self.user_sequential_model.get_user_behavior_embedded(history_item_id, history_len) # (batch_size, embed_dim)
        loss_mse = F.mse_loss(user_behavior_embedded, user_feature_embedded, reduction='mean')

        concat_item_feature = torch.cat([target_item_embedded.squeeze(), item_feature_embedded.squeeze()],
                                        dim=1)  # (batch_size, 2*embed_dim)
        final_item_embeded = self.item_fc_layer(concat_item_feature)  # (batch_size, embed_dim)

        final_user_embeded = user_feature_embedded

        output = torch.sum(final_user_embeded * final_item_embeded, dim=1, keepdim=True)  # (batch_size, 1)

        return output, loss_mse

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        target_item_embedded = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]

        item_feature_embedded = self.item_feature_dense(item_features)  # (batch_size, embed_dim)
        user_feature_embedded = self.user_feature_dense(user_features)  # (batch_size, embed_dim)

        concat_item_feature = torch.cat([target_item_embedded.squeeze(), item_feature_embedded.squeeze()],
                                        dim=1)  # (batch_size, 2*embed_dim)
        final_item_embeded = self.item_fc_layer(concat_item_feature)  # (batch_size, embed_dim)

        final_user_embeded = user_feature_embedded

        output = torch.sum(final_user_embeded * final_item_embeded, dim=1, keepdim=True)  # (batch_size, 1)

        return output
