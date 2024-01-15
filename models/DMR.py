from layer import Embedding

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class deep_match(nn.Module):
    def __init__(self, embed_dim, position_embedd_dim, output_dim):
        super(deep_match, self).__init__()

        self.query_mlp = nn.Sequential(
            nn.Linear(position_embedd_dim, embed_dim),
            nn.PReLU(num_parameters=1, init=0.1)
        )

        self.att_layer = nn.Sequential(
            nn.Linear(embed_dim*4, 80),  # 对应后面的inputs
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.PReLU(num_parameters=1, init=0.1)
        )

    def forward(self, item_his_embedding, context_his_embedding, mask):
        query = context_his_embedding
        query = self.query_mlp(query)

        inputs = torch.cat([query, item_his_embedding, query-item_his_embedding, query*item_his_embedding], dim=-1)  # (batch_size, seq_len, embed_dim*4)

        scores = self.att_layer(inputs)  # (batch_size, seq_len, 1)
        scores = torch.transpose(scores, 1, 2)  # (batch_size, 1, seq_len)

        # mask
        bool_mask = torch.equal(mask, torch.ones_like(mask))
        key_masks = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
        paddings = torch.ones_like(scores) * (-2 ** 32 + 1)
        scores = torch.where(key_masks, scores, paddings)  # (batch_size, 1, seq_len)

        # tril
        scores_tile = torch.tile(torch.sum(scores, dim=1), [1, scores.shape[-1]]) # (batch_size, seq_len*seq_len)
        scores_tile = scores_tile.reshape(scores.shape[0], scores.shape[-1], scores.shape[-1])  # (batch_size, seq_len, seq_len)
        tril = torch.tril(torch.ones_like(scores_tile), diagonal=0)  # (batch_size, seq_len, seq_len)
        paddings = torch.ones_like(scores_tile) * (-2 ** 32 + 1)
        scores_tile = torch.where(tril > 0, scores_tile, paddings)  # (batch_size, seq_len, seq_len)
        scores_tile = torch.softmax(scores_tile, dim=-1)  # (batch_size, seq_len, seq_len)

        att_dm_item_his_embedding = torch.matmul(scores_tile, item_his_embedding)  # (batch_size, seq_len, embed_dim)

        output = self.att_mlp(att_dm_item_his_embedding)  # (batch_size, seq_len, output_dim)

        # target mask
        dm_user_vector = torch.sum(output, dim=1)  # (batch_size, output_dim)

        return dm_user_vector, scores


class DMR(nn.Module):
    def __init__(self, field_dims, embed_dim=4, position_embedd_dim=4):
        super(DMR, self).__init__()
        self.item_embed_dim = embed_dim
        self.position_embedd_dim = position_embedd_dim

        # Embedding
        self.item_embedding = Embedding(field_dims[0], self.item_embed_dim)
        self.position_embedding = Embedding(field_dims[-1], self.position_embedd_dim)
        self.dm_position_embedding = Embedding(field_dims[-1], self.position_embedd_dim)

        # User-to-Item Network
        self.user_to_item_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

