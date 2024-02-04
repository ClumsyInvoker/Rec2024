import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer import Embedding, FullyConnectedLayer


class autodis(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, number, temperature, output_layer=True):
        # input (1*16)->MLP->softmax->(number,1),multiply meta-embedding, output(1*16)
        super().__init__()
        layers = list()
        input_d = input_dim
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())  # try torch.nn.Sigmoid & torch.nn.Tanh
            # layers.append(torch.nn.Sigmoid())
            # layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, number))
        self.mlp = torch.nn.Sequential(*layers)
        self.temperature = temperature

        self.meta_embedding = torch.nn.Parameter(torch.zeros((number, input_d)))  # 20*16
        self.meta_embedding.requires_grad = False
        # torch.nn.init.uniform_(self.meta_embedding, a=-max_val, b=max_val) put into freeze2 3 4

        for param in self.mlp.parameters():
            torch.nn.init.constant_(param.data, 0)
            param.requires_grad = False

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        logits_score = self.mlp(x)  # output(1*20)
        logits_norm_score = torch.nn.Softmax(dim=1)(logits_score / self.temperature)
        autodis_embedding = torch.matmul(logits_norm_score, self.meta_embedding)
        return autodis_embedding


class PLATE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        dim_config = config['dim_config']
        hidden_dim = [32, 32]
        self.max_val = 0.01

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
        # 一阶
        if self.nume_feature_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_feature_size, 1)  # 数值特征的一阶表示
        item_cate_voc_sizes = [voc_size for idx, voc_size in self.item_cate_id_feature_idx]  # 类别特征的词典大小
        item_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.item_cate_one_hot_feature_idx]  # 预留一个缺省embedding
        self.item_fm_1st_order_sparse_emb = nn.ModuleList([
            Embedding(voc_size, 1) for voc_size in item_cate_voc_sizes])  # 类别特征的一阶表示
        user_cate_voc_sizes = [voc_size for idx, voc_size in self.user_cate_id_feature_idx]
        user_cate_voc_sizes += [(end_idx - start_idx + 1 + 1) for start_idx, end_idx in
                                self.user_cate_one_hot_feature_idx]
        self.user_fm_1st_order_sparse_emb = nn.ModuleList([
            Embedding(voc_size, 1) for voc_size in user_cate_voc_sizes])

        # 二阶
        self.item_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in item_cate_voc_sizes])  # 类别特征的二阶表示
        self.user_fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, embed_dim) for voc_size in user_cate_voc_sizes])
        # item id / user id
        self.item_embedding = Embedding(num_embeddings=dim_config['item_id'], embed_dim=embed_dim)
        self.user_embedding = Embedding(num_embeddings=dim_config['user_id'], embed_dim=embed_dim)

        # domain prompt
        self.embedding_prompt = Embedding(num_embeddings=2, embed_dim=embed_dim)

        # user prompt
        self.autodis_model = autodis(input_dim=embed_dim, embed_dims=[32, 32], dropout=0.2, number=10, temperature=0.01)

        self.dense_linear = nn.Linear(self.nume_feature_size, (self.cate_feature_size+4) * embed_dim)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()

        # for DNN
        self.dnn = FullyConnectedLayer(input_size=(self.cate_feature_size+4) * embed_dim,
                                       hidden_size=hidden_dim + [1],
                                       bias=[True, True, False],
                                       batch_norm=True,
                                       dropout_rate=0.5,
                                       activation='relu',
                                       sigmoid=False
                                       )

        # self.sigmoid = nn.Sigmoid() # 使用BCEWithLogitsLoss时不需要sigmoid

    def forward(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold=None):
        """FM 一阶部分"""
        item_emb = self.item_embedding(target_item_id).squeeze(1)  # [bs, embed_dim]
        user_emb = self.user_embedding(user_id).squeeze(1)  # [bs, embed_dim]

        fm_1st_sparse_res = []
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            fm_1st_sparse_res.append(self.item_fm_1st_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float() # [bs, end_idx - start_idx + 1 + 1]
            fm_1st_sparse_res.append(torch.mm(cate_feature, self.item_fm_1st_order_sparse_emb[i].weight))
            i += 1

        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            fm_1st_sparse_res.append(self.user_fm_1st_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_1st_sparse_res.append(torch.mm(cate_feature, self.user_fm_1st_order_sparse_emb[i].weight))
            i += 1

        if len(fm_1st_sparse_res) == 0:
            fm_1st_sparse_res = torch.zeros((user_id.shape[0], 1)).to(user_id.device)
        else:
            fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_feature_size]
            fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]

        if self.nume_feature_size != 0:
            dense_feature = []
            for idx in self.item_nume_feature_idx:
                dense_feature.append(item_features[:, idx].unsqueeze(1))
            for idx in self.user_nume_feature_idx:
                dense_feature.append(user_features[:, idx].unsqueeze(1))
            dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
            fm_1st_dense_res = self.fm_1st_order_dense(dense_feature)
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]

        """FM 二阶部分"""
        fm_2nd_sparse_res = []
        # item类别特征处理
        i = 0
        for idx, voc_size in self.item_cate_id_feature_idx:
            fm_2nd_sparse_res.append(self.item_fm_2nd_order_sparse_emb[i](item_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.item_cate_one_hot_feature_idx:
            cate_feature = item_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_2nd_sparse_res.append(torch.mm(cate_feature, self.item_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        # user类别特征处理
        i = 0
        for idx, voc_size in self.user_cate_id_feature_idx:
            fm_2nd_sparse_res.append(self.user_fm_2nd_order_sparse_emb[i](user_features[:, idx].long()))
            i += 1
        for start_idx, end_idx in self.user_cate_one_hot_feature_idx:
            cate_feature = user_features[:, start_idx: end_idx+1].long()
            if len(cate_feature.shape) == 1:
                cate_feature = cate_feature.unsqueeze(1)
            padding = torch.zeros((cate_feature.shape[0], 1), dtype=torch.long).to(cate_feature.device)
            cate_feature = torch.cat([padding, cate_feature], dim=1).float()  # [bs, end_idx - start_idx + 1 + 1]
            fm_2nd_sparse_res.append(torch.mm(cate_feature, self.user_fm_2nd_order_sparse_emb[i].weight))
            i += 1

        fm_2nd_sparse_res.append(item_emb)
        fm_2nd_sparse_res.append(user_emb)
        fm_2nd_concat_1d = torch.stack(fm_2nd_sparse_res, dim=1)  # [bs, cate_feature_size+2, embed_dim]

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, embed_dim]
        square_sum_embed = sum_embed * sum_embed  # [bs, embed_dim]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, cate_feature_size+2, embed_dim]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, embed_dim]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, embed_dim]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        if is_cold is not None:
            assert isinstance(is_cold, torch.Tensor)
            assert is_cold.shape[0] == user_id.shape[0]
            domain_prompt = self.embedding_prompt(is_cold)
        else:
            default_domain = torch.zeros((user_id.shape[0], 1), dtype=torch.long).to(user_id.device)
            domain_prompt = self.embedding_prompt(default_domain)# [1, embed_dim]

        user_prompt = self.autodis_model(user_emb)  # [bs, embed_dim]

        dnn_input = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, (cate_feature_size+2) * embed_dim]
        dnn_input = torch.cat([domain_prompt.squeeze(1), user_prompt, dnn_input], dim=-1)
        dnn_out = dnn_input

        if self.nume_feature_size != 0:
            dense_feature = []
            for idx in self.item_nume_feature_idx:
                dense_feature.append(item_features[:, idx].unsqueeze(1))
            for idx in self.user_nume_feature_idx:
                dense_feature.append(user_features[:, idx].unsqueeze(1))
            dense_feature = torch.cat(dense_feature, dim=1)  # [bs, nume_feature_size]
            dense_out = self.relu(self.dense_linear(dense_feature))  # [bs, (cate_feature_size+2) * emb_size]
            dnn_out = dnn_out + dense_out  # [bs, (cate_feature_size+2) * emb_size]

        dnn_out = self.dnn(dnn_out)  # [bs, 1]

        out = fm_1st_part + fm_2nd_part + dnn_out  # [bs, 1]
        # out = self.sigmoid(out)
        return out

    def predict(self, user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold=None):
        return self.forward(user_id, target_item_id, history_item_id, history_len, user_features, item_features, is_cold)

    def freeze_when_pretrain(self):
        self.embedding_prompt.weight.requires_grad = False
        # for param in self.parameters():
        #    param.requires_grad = True
        for name, param in self.named_parameters():
            if "autodis_model.meta_embedding" in name:
                torch.nn.init.uniform_(param, a=-self.max_val, b=self.max_val)

    def freeze_when_prompt_tunning(self): # tune prompt
        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            # print(name)
            if "autodis_model.mlp" in name:
                param.requires_grad = True
            if "autodis_model.meta_embedding" in name:
                param.requires_grad = True
                # torch.nn.init.uniform_(param, a=-self.max_val, b=self.max_val)

        self.embedding_prompt.weight.requires_grad = True

