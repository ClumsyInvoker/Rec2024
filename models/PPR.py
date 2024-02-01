import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer
from models.SASRec import SASRec


class PPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        self.prompt_length = 1

        # embedding layers
        self.item_feature_dense = FullyConnectedLayer(input_size=dim_config['item_feature'],
                                                      hidden_size=[embed_dim],
                                                      bias=[True],
                                                      activation='sigmoid')
        self.user_feature_dense = FullyConnectedLayer(input_size=dim_config['user_feature'],
                                                      hidden_size=[embed_dim],
                                                      bias=[True],
                                                      activation='sigmoid')
        self.item_embedding = Embedding(num_embeddings=dim_config['item_id'], embed_dim=embed_dim)
        self.user_embedding = Embedding(num_embeddings=dim_config['user_id'], embed_dim=embed_dim)

        self.user_sequential_model = SASRec(config)
        self.user_personalized_prompt_layer = FullyConnectedLayer(input_size=embed_dim,
                                            hidden_size=[embed_dim, self.prompt_length*embed_dim],
                                            bias=[True, True],
                                            activation='relu',
                                            sigmoid=False)

        self.item_fc_layer = FullyConnectedLayer(input_size=2*embed_dim,
                                            hidden_size=[embed_dim],
                                            bias=[False],
                                            activation='relu',
                                            sigmoid=False) # 如果是BCEWithLogitsLoss， 最后一层不需要sigmoid
        self.user_fc_layer = FullyConnectedLayer(input_size=3*embed_dim,
                                            hidden_size=[embed_dim],
                                            bias=[False],
                                            activation='relu',
                                            sigmoid=False)


    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        # embedding
        user_embedded = self.user_embedding(user_id) # (batch_size, embed_dim)
        target_item_embedded = self.item_embedding(target_item_id) # (batch_size, embed_dim)
        item_feature_embedded = self.item_feature_dense(item_features) # (batch_size, embed_dim)
        user_feature_embedded = self.user_feature_dense(user_features) # (batch_size, embed_dim)

        # prompt
        user_prompt = self.user_personalized_prompt_layer(user_embedded.squeeze(1))
        user_prompt = user_prompt.view(-1, self.prompt_length, self.config['embed_dim']) # (batch_size, prompt_length, embed_dim)
        user_behavior_embedded = self.user_sequential_model.get_user_behavior_embedded_PPR(history_item_id,
                                                                                       history_len, user_prompt)  # (batch_size, embed_dim)

        # concat
        concat_user_feature = torch.cat([user_embedded.squeeze(), user_feature_embedded.squeeze(), user_behavior_embedded], dim=1) # (batch_size, 3*embed_dim)
        final_user_embeded = self.user_fc_layer(concat_user_feature) # (batch_size, embed_dim)
        concat_item_feature = torch.cat([target_item_embedded.squeeze(), item_feature_embedded.squeeze()], dim=1) # (batch_size, 2*embed_dim)
        final_item_embeded = self.item_fc_layer(concat_item_feature) # (batch_size, embed_dim)

        # simple inner product
        output = torch.sum(final_user_embeded * final_item_embeded, dim=1, keepdim=True) # (batch_size, 1)

        return output

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features)

    def load_and_freeze_backbone(self, path, freeze=True):
        # 加载 user_sequential_model
        self.user_sequential_model.load_state_dict(torch.load(path))
        if freeze:
            for param in self.user_sequential_model.parameters():
                param.requires_grad = False