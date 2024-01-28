import torch
import torch.nn as nn

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer

class DeepInterestNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']

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

        self.attn = AttentionSequencePoolingLayer(embed_dim=embed_dim)
        self.fc_layer = FullyConnectedLayer(input_size=3*embed_dim,
                                            hidden_size=[200, 80, 1],
                                            bias=[True, True, False],
                                            activation='relu',
                                            sigmoid=True)

    def forward(self, target_item_id, history_item_id, history_len, user_features, item_features):
        # embedding
        target_item_embedded = self.item_embedding(target_item_id) # (batch_size, embed_dim)
        history_item_embedded = self.item_embedding(history_item_id) # (batch_size, max_history_len, embed_dim)
        item_feature_embedded = self.item_feature_dense(item_features) # (batch_size, embed_dim)
        user_feature_embedded = self.user_feature_dense(user_features) # (batch_size, embed_dim)

        # attention
        history = self.attn(target_item_embedded, history_item_embedded, history_len) # (batch_size, 1, embed_dim)

        # concat
        concat_feature = torch.cat([history.squeeze(), item_feature_embedded.squeeze(), user_feature_embedded.squeeze()], dim=1) # (batch_size, 2*embed_dim)

        # fully-connected layers
        output = self.fc_layer(concat_feature)

        return output
