import numpy as np
import torch
import torch.nn as nn

from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate=0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # input shape: (batch_size, max_lens, embed_dim), output shape: (batch_size, max_lens, embed_dim)
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        maxlen = config['maxlen']
        num_blocks = 2
        num_heads = 4
        dropout_rate = 0.1

        self.user_num = dim_config['user_id']
        self.item_num = dim_config['item_id']

        self.item_embedding = Embedding(self.item_num, embed_dim)
        self.position_embedding = Embedding(maxlen, embed_dim)
        self.embedding_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(embed_dim, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(embed_dim, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(embed_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(embed_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs, mask):
        device = self.config['device']
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.position_embedding(torch.LongTensor(positions).to(device))
        seqs = self.embedding_dropout(seqs) # (batch_size, max_lens, embed_dim)

        timeline_mask = mask.unsqueeze(-1) # (batch_size, max_lens) -> (batch_size, max_lens, 1)
        seqs *= timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask

        log_feats = self.last_layernorm(seqs) # (batch_size, max_lens, embed_dim)


        return log_feats

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        user_behavior_embedded = self.log2feats(history_item_id, history_len) # (batch_size, max_lens, embed_dim)

        mask = history_len.bool().unsqueeze(-1) # (batch_size, max_history_len, 1)
        mask = torch.tile(mask, [1, 1, user_behavior_embedded.shape[-1]]) # (batch_size, max_history_len, embed_dim)
        user_behavior_embedded = torch.where(mask, user_behavior_embedded, torch.zeros_like(user_behavior_embedded))
        user_behavior_embedded = torch.sum(user_behavior_embedded, dim=1) / torch.sum(history_len, dim=1, keepdim=True) # (batch_size, embed_dim)
        user_behavior_embedded = torch.where(torch.isnan(user_behavior_embedded), torch.zeros_like(user_behavior_embedded), user_behavior_embedded)
        final_user_embedding = user_behavior_embedded

        final_item_embedding = self.item_embedding(target_item_id).squeeze(1) # (batch_size, embed_dim)

        logits = (final_user_embedding * final_item_embedding).sum(dim=-1, keepdim=True)

        return logits

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features)

    def get_user_behavior_embedded(self, history_item_id, history_len):
        user_behavior_embedded = self.log2feats(history_item_id, history_len) # (batch_size, max_lens, embed_dim)
        mask = history_len.bool().unsqueeze(-1)  # (batch_size, max_history_len, 1)
        mask = torch.tile(mask, [1, 1, user_behavior_embedded.shape[-1]])  # (batch_size, max_history_len, embed_dim)
        user_behavior_embedded = torch.where(mask, user_behavior_embedded, torch.zeros_like(user_behavior_embedded))
        user_behavior_embedded = torch.sum(user_behavior_embedded, dim=1) / torch.sum(history_len, dim=1,
                                                                                      keepdim=True)  # (batch_size, embed_dim)
        user_behavior_embedded = torch.where(torch.isnan(user_behavior_embedded),
                                             torch.zeros_like(user_behavior_embedded), user_behavior_embedded)

        return user_behavior_embedded


    def log2feats_PPR(self, log_seqs, mask, user_prompt):
        device = self.config['device']
        prompt_length = user_prompt.shape[1]

        seqs = self.item_embedding(log_seqs)
        seqs[:, :prompt_length, :] = user_prompt # 将最前面的替换成user_prompt
        seqs *= self.item_embedding.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.position_embedding(torch.LongTensor(positions).to(device))
        seqs = self.embedding_dropout(seqs) # (batch_size, max_lens, embed_dim)

        timeline_mask = mask.unsqueeze(-1) # (batch_size, max_lens) -> (batch_size, max_lens, 1)
        seqs *= timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask

        log_feats = self.last_layernorm(seqs) # (batch_size, max_lens, embed_dim)
        return log_feats


    def get_user_behavior_embedded_PPR(self, history_item_id, history_len, user_prompt):
        user_behavior_embedded = self.log2feats_PPR(history_item_id, history_len, user_prompt) # (batch_size, max_lens, embed_dim)\
        mask = history_len.bool().unsqueeze(-1)  # (batch_size, max_history_len, 1)
        mask = torch.tile(mask, [1, 1, user_behavior_embedded.shape[-1]])  # (batch_size, max_history_len, embed_dim)
        user_behavior_embedded = torch.where(mask, user_behavior_embedded, torch.zeros_like(user_behavior_embedded))
        user_behavior_embedded = torch.sum(user_behavior_embedded, dim=1) / torch.sum(history_len, dim=1,
                                                                                      keepdim=True)  # (batch_size, embed_dim)
        user_behavior_embedded = torch.where(torch.isnan(user_behavior_embedded),
                                             torch.zeros_like(user_behavior_embedded), user_behavior_embedded)

        return user_behavior_embedded