import pickle
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import array
from torch.nn import Linear

class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, parameterization='matrix', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])
        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  
                dot_ = xl_w + self.bias[i]  
                x_l = x_0 * dot_ + x_l 
            else:  
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class GNN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(GNN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size
        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.mode = args.mode
        self.include_feature = args.include_feature
        self.include_feature_pretrain = False
        self.embedding_dict, self.weight_dict = self.init_weight()

        if args.include_feature == True:
            self.item_raw_feature = torch.from_numpy(pd.read_csv('item_info.csv',header=None).values).float().to(self.device)
            self.user_raw_feature = torch.from_numpy(pd.read_csv('user_info.csv',header=None).values).float().to(self.device)
            self.id_embedding_dict , self.embedding_dict = self.init_weight_fea()

            self.item_fea_tran = Linear(self.item_raw_feature.shape[1], self.emb_size).to(self.device)
            self.item_fea_cro = CrossNet(self.emb_size).to(self.device)
            self.user_fea_tran = Linear(self.user_raw_feature.shape[1], self.emb_size).to(self.device)
            self.user_fea_cro = CrossNet(self.emb_size).to(self.device)

            self.att1 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.att2 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.att3 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.att4 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.att5 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.att6 = torch.nn.Linear(in_features=self.emb_size, out_features=1).to(self.device)
            self.get_feature()

        if self.include_feature_pretrain == True:
            self.embedding_dict['item_emb'] = pickle.load(open("{}/item_emb_pretrain.pkl".format('.'), "rb"))
            self.embedding_dict['user_emb'] = pickle.load(open("{}/user_emb_pretrain.pkl".format('.'), "rb"))


        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        return embedding_dict, weight_dict

    def init_weight_fea(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        id_embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                             self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                             self.emb_size)))
        })

        initializer2 = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer2(torch.empty(self.n_user,
                                                             self.emb_size * 4))),
            'item_emb': nn.Parameter(initializer2(torch.empty(self.n_item,
                                                             self.emb_size * 4)))
        })
        return id_embedding_dict, embedding_dict

    def get_weight(self):
        return self.embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def get_u_i_embedding(self, users, pos_items):
        return self.embedding_dict['user_emb'][users],self.embedding_dict['item_emb'][pos_items]

    def get_feature(self):
        item_raw_feature = self.item_raw_feature.to(self.device)
        item_fea_emb = self.item_fea_tran(item_raw_feature)
        item_fc_emb = self.item_fea_cro(item_fea_emb)

        user_raw_feature = self.user_raw_feature.to(self.device)
        user_fea_emb = self.user_fea_tran(user_raw_feature)
        user_fc_emb = self.user_fea_cro(user_fea_emb)

        att1 = torch.exp(torch.tanh(self.att1(self.id_embedding_dict['item_emb'].to(self.device))))
        att2 = torch.exp(torch.tanh(self.att2(item_fea_emb)))
        att3 = torch.exp(torch.tanh(self.att3(item_fc_emb)))

        att1_ = att1 / (att1 + att2 + att3)
        att2_ = att2 / (att1 + att2 + att3)
        att3_ = att3 / (att1 + att2 + att3)

        att1_ = att1_.repeat(  (1, self.emb_size) )
        att2_ = att2_.repeat((1, self.emb_size))
        att3_ = att3_.repeat((1, self.emb_size))

        att_embedding1 = torch.mul(att1_, self.id_embedding_dict['item_emb'].to(self.device)).squeeze()
        att_embedding2 = torch.mul(att2_, item_fea_emb).squeeze()
        att_embedding3 = torch.mul(att3_, item_fc_emb).squeeze()

        item_att_embedding = att_embedding1 + att_embedding2 + att_embedding3
        self.embedding_dict['item_emb'] = nn.Parameter(item_att_embedding)

        att4 = torch.exp(torch.tanh(self.att4(self.id_embedding_dict['user_emb'].to(self.device))))
        att5 = torch.exp(torch.tanh(self.att5(user_fea_emb)))
        att6 = torch.exp(torch.tanh(self.att6(user_fc_emb)))

        att4_ = att4 / (att4 + att5 + att6)
        att5_ = att5 / (att4 + att5 + att6)
        att6_ = att6 / (att4 + att5 + att6)

        att4_ = att4_.repeat((1, self.emb_size))
        att5_ = att5_.repeat((1, self.emb_size))
        att6_ = att6_.repeat((1, self.emb_size))

        att_embedding1 = torch.mul(att4_, self.id_embedding_dict['user_emb'].to(self.device)).squeeze()
        att_embedding2 = torch.mul(att5_, user_fea_emb).squeeze()
        att_embedding3 = torch.mul(att6_, user_fc_emb).squeeze()

        user_att_embedding = att_embedding1 + att_embedding2 + att_embedding3
        self.embedding_dict['user_emb'] = nn.Parameter(user_att_embedding)
        return 


    def forward(self, users, pos_items, neg_items, drop_flag=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        if not self.mode=='central':
            mask_mat = torch.zeros(size=A_hat.shape,device='cuda')
            for i in range(len(users)):
                mask_mat[users[0],:] = 1.
                mask_mat[:,users[0]] = 1.
                mask_mat[self.n_user + pos_items[i], :] = 1.
                mask_mat[:, self.n_user + pos_items[i]] = 1.
            A_hat = torch.mul(A_hat.to_dense(), mask_mat)

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        embs = all_embeddings

        a=all_embeddings

        for layer in range(len(self.layers)):
            all_emb = torch.mm(A_hat, ego_embeddings)
            embs.append(all_emb)

        embs = torch.stack(embs)

        light_out = torch.mean(embs, dim=0)

        all_embeddings = torch.cat(all_embeddings, 1)

        u_g_embeddings = light_out[:self.n_user, :]
        i_g_embeddings = light_out[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
