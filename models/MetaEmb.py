import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer
from models.DeepFM import DeepFM

class MetaEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']

        self.item_cate_id_feature_idx = config['item_feature']['cate_id_feat_idx']  # 类别id特征
        self.item_cate_one_hot_feature_idx = config['item_feature']['cate_one_hot_feat_idx']  # 类别one-hot特征
        self.item_cate_feature_size = len(self.item_cate_id_feature_idx) + len(self.item_cate_one_hot_feature_idx)

        self.deepfm = DeepFM(config)
        self.meta_emb = nn.Sequential(
            nn.Linear(self.item_cate_feature_size * embed_dim, embed_dim, bias=False),
            nn.Tanh()
        )

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold_item,
                item_meta_emb=None):
        out = self.deepfm(user_id, target_item_id, history_item_id, history_len, user_features, item_features)

        if item_meta_emb is None:
            item_features_emb = self.deepfm.get_item_feature_embedding(item_features)
            item_meta_emb = self.meta_emb(item_features_emb)
        else:
            item_meta_emb = item_meta_emb

        out_cold = self.deepfm.get_logit_MetaEmb(user_id, item_meta_emb, user_features, item_features)

        final_out = torch.where(is_cold_item.bool(), out_cold, out)

        return final_out

    def get_logit_and_meta_emb(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold_item,
                item_meta_emb=None):
        out = self.deepfm(user_id, target_item_id, history_item_id, history_len, user_features, item_features)

        if item_meta_emb is None:
            item_features_emb = self.deepfm.get_item_feature_embedding(item_features)
            item_meta_emb = self.meta_emb(item_features_emb)
        else:
            item_meta_emb = item_meta_emb

        out_cold = self.deepfm.get_logit_MetaEmb(user_id, item_meta_emb, user_features, item_features)

        final_out = torch.where(is_cold_item.bool(), out_cold, out)

        return final_out, item_meta_emb

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold_item):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold_item)

    def get_item_meta_emb(self, item_features):
        item_features_emb = self.deepfm.get_item_feature_embedding(item_features)
        item_meta_emb = self.meta_emb(item_features_emb)
        return item_meta_emb

    def load_deepfm_and_freeze(self, deepfm_model_path, freeze=True):
        self.deepfm.load_state_dict(torch.load(deepfm_model_path, map_location=torch.device(self.config['device'])))
        if freeze:
            for param in self.deepfm.parameters():
                param.requires_grad = False

