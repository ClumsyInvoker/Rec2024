from layer import Embedding

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class deep_match(nn.Module):
    def __init__(self, embed_dim, position_embedd_dim, output_dim):
        super().__init__()

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


class dmr_fcn_attention(nn.Module):
    def __init__(self, embed_dim, position_embedd_dim, output_dim):
        super(dmr_fcn_attention, self).__init__()

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

    def forward(self, item_embedding, item_his_embedding, context_his_embedding, mask, mode="SUM"):
        mask = torch.equal(mask, torch.ones_like(mask))
        item_embedding_tile = torch.tile(item_embedding, [1, mask.shape[1]])  # (batch_size, seq_len*embed_dim)
        item_embedding_tile = item_embedding_tile.reshape(-1, mask.shape[1], item_embedding.shape[-1])  # (batch_size, seq_len, embed_dim)

        if context_his_embedding is not None:
            query = torch.cat([item_embedding_tile, context_his_embedding], dim=-1)
        else:
            query = item_embedding_tile
        query = self.query_mlp(query)
        dmr_all = torch.cat([query, item_his_embedding, query-item_his_embedding, query*item_his_embedding], dim=-1)  # (batch_size, seq_len, embed_dim*4)
        scores = self.att_layer(dmr_all)  # (batch_size, seq_len, 1)
        scores = torch.transpose(scores, 1, 2)  # (batch_size, 1, seq_len)

        # mask
        key_masks = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
        paddings = torch.ones_like(scores) * (-2 ** 32 + 1)
        padding_no_softmax = torch.zeros_like(scores)
        scores = torch.where(key_masks, scores, paddings)  # (batch_size, 1, seq_len)
        scores_no_softmax = torch.where(key_masks, scores, padding_no_softmax)  # (batch_size, 1, seq_len)

        scores = torch.softmax(scores, dim=-1)  # (batch_size, 1, seq_len)

        if mode == "SUM":
            output = torch.matmul(scores, item_his_embedding) # (batch_size, 1, embed_dim)
            output = torch.sum(output, dim=1)  # (batch_size, embed_dim)
        else:
            scores = torch.reshape(scores, [-1, item_his_embedding.shape[1]])  # (batch_size, seq_len)
            output = item_his_embedding * scores.unsqueeze(-1)  # (batch_size, seq_len, embed_dim)
            output = torch.reshape(output, item_his_embedding.shape)  # (batch_size, seq_len, embed_dim)

        return output, scores, scores_no_softmax


class DMR(nn.Module):
    def __init__(self, config):
        super(DMR, self).__init__()
        field_dims = config['field_dims']
        embed_dim = config['embed_dim']
        position_embedd_dim = config['position_embedd_dim']

        self.item_embed_dim = embed_dim
        self.position_embedd_dim = position_embedd_dim

        # Embedding
        self.item_embedding = Embedding(field_dims[0], self.item_embed_dim)
        self.position_embedding = Embedding(field_dims[-1], self.position_embedd_dim)
        self.dm_position_embedding = Embedding(field_dims[-1], self.position_embedd_dim)

        # User-to-Item Network
        self.user_to_item_net = deep_match(embed_dim, position_embedd_dim, embed_dim)
        # Item-to-Item Network
        self.item_to_item_net = dmr_fcn_attention(embed_dim, position_embedd_dim, embed_dim)

    def forward(self, item, item_his, mask):
        item_embedding = self.item_embedding(item) # (batch_size, embed_dim)
        item_his_embedding = self.item_embedding(item_his) # (batch_size, seq_len, embed_dim)

        position = torch.arange(item_his.shape[1], dtype=torch.long, device=item_his.device)
        position_embedding = self.position_embedding(position) # (seq_len, position_embedd_dim)
        position_embedding = torch.tile(position_embedding, [item_his.shape[0], 1])  # (batch_size*seq_len, position_embedd_dim)
        position_embedding = torch.reshape(position_embedding, [item_his.shape[0], item_his.shape[1], -1])  # (batch_size, seq_len, position_embedd_dim)

        dm_position = torch.arange(item_his.shape[1], dtype=torch.long, device=item_his.device)
        dm_position_embedding = self.dm_position_embedding(dm_position) # (seq_len, position_embedd_dim)
        dm_position_embedding = torch.tile(dm_position_embedding, [item_his.shape[0], 1])  # (batch_size*seq_len, position_embedd_dim)
        dm_position_embedding = torch.reshape(dm_position_embedding, [item_his.shape[0], item_his.shape[1], -1])  # (batch_size, seq_len, position_embedd_dim)

        dm_user_vector, dm_scores = self.user_to_item_net(item_his_embedding, dm_position_embedding, mask)  # (batch_size, embed_dim), (batch_size, 1, seq_len)

        att_outputs, alphas, scores_unnorm = self.item_to_item_net(item_embedding, item_his_embedding, position_embedding, mask, mode="SUM")  # (batch_size, embed_dim), (batch_size, 1, seq_len), (batch_size, 1, seq_len)




