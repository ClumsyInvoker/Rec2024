import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DSSM import DSSM
from models.layer import Embedding, FullyConnectedLayer, AttentionSequencePoolingLayer


class DSSM_PTCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        self.prompt_embed_dim = prompt_embed_dim =  config['prompt_embed_dim']
        self.prompt_net_hidden_size = prompt_net_hidden_size =  config['prompt_net_hidden_size']

        self.backbone = DSSM(config)
        prompt_net_total_size = embed_dim * prompt_net_hidden_size + prompt_net_hidden_size + \
                                + prompt_net_hidden_size * prompt_embed_dim + prompt_embed_dim
        self.prompt_item_embedding = Embedding(num_embeddings=dim_config['item_id'], embed_dim=prompt_embed_dim)
        self.prompt_user_embedding = Embedding(num_embeddings=dim_config['user_id'], embed_dim=prompt_embed_dim)
        self.prompt_generator = FullyConnectedLayer(input_size=embed_dim,
                                                      hidden_size=[prompt_embed_dim+prompt_net_total_size],
                                                      bias=[True],
                                                      activation='relu')

        self.fusion_layer = FullyConnectedLayer(input_size=embed_dim+prompt_embed_dim*2,
                                            hidden_size=[embed_dim],
                                            bias=[False],
                                            activation='relu',
                                            sigmoid=False)

        self.loss_pfpe = nn.Softplus(beta=1, threshold=10)

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features,
                     item_pos_feedback, item_pos_feedback_mask, item_neg_feedback, item_neg_feedback_mask
                ):
        final_user_embed, final_item_embed = self.backbone.get_final_user_embed(user_id, target_item_id, history_item_id, history_len, user_features, item_features)
        # prompt_input = item_pos_feedback_embed
        prompt_input = self.prompt_item_embedding(target_item_id.squeeze()) # (batch_size, prompt_embed_dim)
        total_prompt = self.prompt_generator(prompt_input) # (batch_size, prompt_embed_dim+prompt_net_total_size)
        # print(total_prompt.shape)
        prompt_embed = total_prompt[:, :self.prompt_embed_dim] # (batch_size, prompt_embed_dim)

        prompt_net = total_prompt[:, self.prompt_embed_dim:] # (batch_size, prompt_net_total_size)

        pos_feedback_embed = self.prompt_user_embedding(item_pos_feedback)
        pos_prompt_embed = self.get_final_prompt_emebed(pos_feedback_embed, prompt_net) # (batch_size, max_feedback_len, prompt_embed_dim)
        pos_prompt_embed = torch.where(item_pos_feedback_mask.unsqueeze(-1).bool(), pos_prompt_embed, torch.zeros_like(pos_prompt_embed))

        neg_feedback_embed = self.prompt_user_embedding(item_neg_feedback)
        neg_prompt_embed = self.get_final_prompt_emebed(neg_feedback_embed, prompt_net) # (batch_size, max_feedback_len, prompt_embed_dim)
        neg_prompt_embed = torch.where(item_neg_feedback_mask.unsqueeze(-1).bool(), neg_prompt_embed, torch.zeros_like(neg_prompt_embed))

        final_pos_prompt_embed = torch.sum(pos_prompt_embed, dim=1) # (batch_size, prompt_embed_dim)
        final_neg_prompt_embed = torch.sum(neg_prompt_embed, dim=1) # (batch_size, prompt_embed_dim)
        pos_num = torch.sum(item_pos_feedback_mask, dim=1, keepdim=True) # (batch_size, 1)
        neg_num = torch.sum(item_neg_feedback_mask, dim=1, keepdim=True) # (batch_size, 1)
        loss_pfpe = self.loss_pfpe(-(final_pos_prompt_embed * neg_num - final_neg_prompt_embed * pos_num)) # (batch_size, 1)

        fusion_input = torch.cat([final_item_embed, final_pos_prompt_embed, prompt_embed], dim=1) # (batch_size, embed_dim+prompt_embed_dim*2)
        final_item_embed = self.fusion_layer(fusion_input) # (batch_size, embed_dim)

        # simple inner product
        output = torch.sum(final_user_embed * final_item_embed, dim=1, keepdim=True) # (batch_size, 1)

        return output, loss_pfpe

    def get_final_prompt_emebed(self, user_embed, prompt_net):
        # user_embed: (batch_size, len, embed_dim)
        pos = 0
        prompt_net_layer1_w = prompt_net[:, pos:pos+self.embed_dim * self.prompt_net_hidden_size]\
            .reshape(-1, self.embed_dim, self.prompt_net_hidden_size) # (batch_size, embed_dim, prompt_net_hidden_size)
        pos += self.embed_dim * self.prompt_net_hidden_size
        prompt_net_layer1_b = prompt_net[:, pos:pos+self.prompt_net_hidden_size] # (batch_size, prompt_net_hidden_size)
        pos += self.prompt_net_hidden_size
        prompt_net_layer2_w = prompt_net[:, pos:pos+self.prompt_net_hidden_size * self.prompt_embed_dim]\
            .reshape(-1, self.prompt_net_hidden_size, self.prompt_embed_dim) # (batch_size, prompt_net_hidden_size, prompt_embed_dim)
        pos += self.prompt_net_hidden_size * self.prompt_embed_dim
        prompt_net_layer2_b = prompt_net[:, pos:pos+self.prompt_embed_dim] # (batch_size, prompt_embed_dim)

        # print(user_embed.shape, prompt_net_layer1_w.shape, prompt_net_layer1_b.shape)
        output = torch.matmul(user_embed, prompt_net_layer1_w) + prompt_net_layer1_b.unsqueeze(1) # (batch_size, len, prompt_net_hidden_size)
        output = torch.relu(output)
        # print(user_embed.shape, prompt_net_layer2_w.shape, prompt_net_layer2_b.shape)
        output = torch.matmul(output, prompt_net_layer2_w) + prompt_net_layer2_b.unsqueeze(1) # (batch_size, len, prompt_embed_dim)

        return output

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features,
                     item_pos_feedback, item_pos_feedback_mask):
        final_user_embed, final_item_embed = self.backbone.get_final_user_embed(user_id, target_item_id,
                                                                                history_item_id, history_len,
                                                                                user_features, item_features)

        prompt_input = self.prompt_item_embedding(target_item_id.squeeze())  # (batch_size, prompt_embed_dim)
        total_prompt = self.prompt_generator(prompt_input)  # (batch_size, prompt_embed_dim+prompt_net_total_size)
        # print(total_prompt.shape)
        prompt_embed = total_prompt[:, :self.prompt_embed_dim]  # (batch_size, prompt_embed_dim)

        prompt_net = total_prompt[:, self.prompt_embed_dim:]  # (batch_size, prompt_net_total_size)

        pos_feedback_embed = self.prompt_user_embedding(item_pos_feedback)
        pos_prompt_embed = self.get_final_prompt_emebed(pos_feedback_embed,
                                                        prompt_net)  # (batch_size, max_feedback_len, prompt_embed_dim)
        pos_prompt_embed = torch.where(item_pos_feedback_mask.unsqueeze(-1).bool(), pos_prompt_embed,
                                       torch.zeros_like(pos_prompt_embed))

        final_pos_prompt_embed = torch.sum(pos_prompt_embed, dim=1) # (batch_size, prompt_embed_dim)

        fusion_input = torch.cat([final_item_embed, final_pos_prompt_embed, prompt_embed],
                                 dim=1)  # (batch_size, embed_dim+prompt_embed_dim*2)
        final_item_embed = self.fusion_layer(fusion_input)  # (batch_size, embed_dim)

        # simple inner product
        output = torch.sum(final_user_embed * final_item_embed, dim=1, keepdim=True)  # (batch_size, 1)

        return output


    def load_and_freeze_backbone(self, path):
        self.backbone.load_state_dict(torch.load(path))
        for param in self.backbone.parameters():
            param.requires_grad = False
