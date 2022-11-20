import copy
import pickle
import random
import sys
import math
from statistics import mean
import numpy as np
import torch
import torch.optim as optim
from numpy import array
from GNN import GNN
from utility.helper import *
from utility.batch_test import *
import warnings
# warnings.filterwarnings('ignore')
from time import time
from sklearn.cluster import KMeans

def FedWeiAvg(w,w_):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], w_[0])
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k] * w_[i])
    return w_avg

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

if __name__ == '__main__':

    n_fed_client_each_round = 128
    n_cluster = 5
    args.device = torch.device('cuda')
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.mode = 'fed'
    args.include_feature = False
    n_client = data_generator.n_users
    model = GNN(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    idxs_users = random.sample(range(0, n_client), n_fed_client_each_round)
    wei_usrs = [1./n_fed_client_each_round] * n_fed_client_each_round
    best_hr_10, best_hr_20, best_ndcg_10, best_ndcg_20 = 0, 0, 0, 0
    training_time = 0.0,
    begin_time = time()

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        model_para_list = []
        user_ini_state = copy.deepcopy(model.state_dict())['embedding_dict.user_emb']
        user_emb_list = {}
        local_test = []
        w_cluster =  {i:[] for i in range(n_cluster)}
        local_model_test_res = {}
        def LocalTestNeg(data_generator,test_idx,model):
            test_positive, test_nagetive = data_generator.sample_test_nagative(test_idx)
            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx], test_positive)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()[0]
            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx] * len(test_nagetive),
                                                                         test_nagetive)
            rate_batch_nagetive = torch.matmul(u_g_embeddings.unsqueeze(1),
                                               pos_i_g_embeddings.unsqueeze(2)).squeeze().detach().cpu()
            torch_cat = torch.cat((rate_batch, rate_batch_nagetive), 0).numpy()
            return torch_cat

        for idx in (idxs_users):
            if epoch>1:
                model.load_state_dict(copy.deepcopy(w_cluster_avg[clu_result[idx]]))
                user_ini_state = copy.deepcopy(model.state_dict())['embedding_dict.user_emb']

            model_ini = copy.deepcopy(model.state_dict())
            users, pos_items, neg_items = data_generator.sample(idx)
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            local_model_test_res[idx] = LocalTestNeg(data_generator,idx,model)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            model_aft = copy.deepcopy(model.state_dict())
            for _ in range(100):
                ran_idx = random.randint(0, data_generator.n_items-1)
                loc, scale = 0., 0.02
                s = torch.Tensor(np.random.laplace(loc, scale, args.embed_size)).to('cuda')
                model_aft['embedding_dict.item_emb'][ran_idx] = model_aft['embedding_dict.item_emb'][ran_idx] + s
            model_para_list += [model_aft]

            user_emb_list[idx] = model_aft['embedding_dict.user_emb'][idx]

            if epoch >= 1:
                w_cluster[clu_result[idx] ].append(model_aft)

            model.load_state_dict(model_ini)

        w_ = FedAvg(model_para_list)

        for j in user_emb_list:
            user_ini_state[j] = user_emb_list[j]

        w_['embedding_dict.user_emb'] = copy.deepcopy(user_ini_state)

        if epoch>=1 :
            w_cluster_avg = []
            for i in range(len(w_cluster)):
                try:
                    w_cluster_avg.append(FedAvg(w_cluster[i]))
                except Exception:
                    w_cluster_avg.append(w_)

        res_list = []
        if epoch > 1:
            print('Cluster test')
            hr_list = []
            ndcg_list = []
            hr_list_20 = []
            ndcg_list_20 = []
            for i in range(len(w_cluster)):
                users_to_test = [x for x in range(n_client) if clu_result[x]==i]

                for test_idx in (users_to_test):
                    model.load_state_dict(w_cluster_avg[i])
                    torch_cat = LocalTestNeg(data_generator, test_idx, model)

                    model.load_state_dict(w_)
                    torch_cat2 = LocalTestNeg(data_generator, test_idx, model)

                    if local_model_test_res.get(test_idx) is not None:
                        torch_cat = np.mean( np.array([ torch_cat2,torch_cat, local_model_test_res[test_idx] ]), axis=0 )
                    else:
                        torch_cat = np.mean(np.array([torch_cat2,torch_cat]), axis=0)

                    ranking = list(np.argsort(torch_cat))[::-1].index(0) + 1

                    ndcg = 0
                    hr = 0
                    if ranking <= 10:
                        hr = 1
                        ndcg = math.log(2) / math.log(1 + ranking)
                    hr_list.append(hr), ndcg_list.append(ndcg)

                    ndcg = 0
                    hr = 0
                    if ranking <= 20:
                        hr = 1
                        ndcg = math.log(2) / math.log(1 + ranking)
                    hr_list_20.append(hr), ndcg_list_20.append(ndcg)
            print(f'HR@10={mean(hr_list)},NDCG@10={mean(ndcg_list)}')

            if mean(hr_list) > best_hr_10 or mean(ndcg_list) > best_ndcg_10 or mean(
                        hr_list_20) > best_hr_20 or mean(
                        ndcg_list_20) > best_ndcg_20:
                    best_hr_10 = max(best_hr_10, mean(hr_list))
                    best_ndcg_10 = max(best_ndcg_10, mean(ndcg_list))
                    best_hr_20 = max(best_hr_10, mean(hr_list_20))
                    best_ndcg_20 = max(best_ndcg_20, mean(ndcg_list_20))
                    training_time = time() - begin_time

        model.load_state_dict(w_)
        users_emb = w_['embedding_dict.user_emb']
        users_emb = users_emb.cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(users_emb)
        clu_result = list(kmeans.labels_)
        clu_result_dict = dict((x, clu_result.count(x)) for x in set(clu_result))
        clu_result_weight = [clu_result_dict[i] for i in clu_result]

        n_rdm = int(n_fed_client_each_round/2)
        n_choice = int(n_fed_client_each_round/2)
        idxs_users_rdm = random.sample(range(0, n_client), n_rdm)
        idxs_users_choice = random.choices(range(0, n_client), weights=(clu_result_weight), k=n_choice)
        idxs_users_ = idxs_users_rdm + idxs_users_choice
        idxs_users_ = list(set(idxs_users_))
        while len(idxs_users_) < n_fed_client_each_round:
            r_= random.sample(range(0, n_client), 1)[0]
            if not r_ in idxs_users_:
                idxs_users_.append(r_)

        idxs_users = idxs_users_
        random.shuffle(idxs_users)

        wei_usrs = [clu_result_weight[i] for i in idxs_users]
        wei_usrs = [i/sum(wei_usrs) for i in wei_usrs]

        if epoch % 10 == 0:
            print(f"Best Result:{'%.4f' % best_hr_10},{'%.4f' % best_ndcg_10},{'%.4f' % best_hr_20},{'%.4f' % best_ndcg_20}")
            print(f'Trainig time:{training_time}')

        if (epoch + 1) % 50 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=True)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        if should_stop == True:
            break

        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    emd_dict = model.get_weight()
