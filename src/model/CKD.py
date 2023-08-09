import sys
import argparse
from src.model.transform_model import *

print(sys.getdefaultencoding())
from datetime import datetime
import math
import argparse
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from src.model.ckdutil import *
from src.model.ckdmodel import *

torch.backends.cudnn.benchmark = True
from sklearn.metrics import roc_auc_score, average_precision_score
from functools import reduce
from tqdm import tqdm

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-dataset', required=True, type=str, help='Targeting dataset.',
#                         choices=['DBLP', 'DBLP2', 'PubMed', 'ACM', 'ACM2', 'Freebase'])
#     parser.add_argument('-model', required=True, type=str, default='CKD')
#     parser.add_argument('-attributed', type=str, default=False)
#     parser.add_argument('-supervised', type=str, default=False)
#     parser.add_argument('-version', type=str, default='link')

    # return parser.parse_args()


def score(criterion, emb_list, graph_emb_list, status_list):
    index = torch.Tensor([0]).long().cpu()
    loss = None
    for idx in range(len(emb_list)):
        emb_list[idx] = emb_list[idx].index_select(dim=1, index=index).squeeze().cpu()
    for idx in range(len(emb_list)):
        for idy in range(len(emb_list)):
            node_emb = emb_list[idx]
            graph_emb = graph_emb_list[idy]
            mask = torch.Tensor([i[idy] for i in status_list]).bool().cpu()
            pos = torch.sum(node_emb * graph_emb, dim=1).squeeze().masked_select(mask)
            matrix = torch.mm(node_emb, graph_emb.T)
            mask_idx = torch.Tensor([i for i in range(len(status_list)) if status_list[i][idy] == 0]).long().cpu()
            neg_mask = np.ones(shape=(node_emb.shape[0], node_emb.shape[0]))
            row, col = np.diag_indices_from(neg_mask)
            neg_mask[row, col] = 0
            neg_mask = torch.from_numpy(neg_mask).bool().cpu()
            neg_mask[mask_idx,] = 0
            neg = matrix.masked_select(neg_mask)

            if pos.shape[0] == 0:
                continue
            if loss is None:
                loss = criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cpu())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cpu())
            else:
                loss += criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cpu())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cpu())
    return loss


def global_score(criterion, emb_list, graph_emb_list, neg_graph_emb_list, status_list):
    loss = None

    for idx in range(len(emb_list)):
        for idy in range(len(emb_list)):
            node_emb = emb_list[idx]
            global_emb = graph_emb_list[idy]
            neg_global_emb = neg_graph_emb_list[idy]
            mask = torch.Tensor([i[idx] for i in status_list]).bool().cpu()
            pos = torch.sum(node_emb * global_emb, dim=1).squeeze().masked_select(mask)
            neg = torch.sum(node_emb * neg_global_emb, dim=1).squeeze().masked_select(mask)
            if pos.shape[0] == 0:
                continue

            if loss is None:
                loss = criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cpu())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cpu())
            else:
                loss += criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cpu())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cpu())
    return loss



def transform(args):

    print('Transforming {} to {} input format for {}, {} training!'
          .format(args.dataset, args.model,
                  'attributed' if args.attributed == 'True' else 'unattributed',
                  'semi-supervised' if args.supervised == 'True' else 'unsupervised'))

    if args.model == 'CKD': ckd_link_convert(args.target_node, args.data_type, args.relation_list, args.dataset, args.attributed, args.version, False)

    print('Data transformation finished!')

    return


def delete_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def CKD(args):
    # transform
    if args.transform:
        transform(args)

    print(f'start time:{datetime.now()}')
    cuda_device = -1
    torch.cuda.set_device(cuda_device)
    torch.backends.cudnn.benchmark = False
    print('cuda:', cuda_device)

    # args = parse_args()
    print(f'emb size:{args.dim}')

    set_seed(args.seed, args.device)
    print(f'seed:{args.seed}')

    print(
        f'dataset:{args.dataset},attributed:{args.attributed},ltypes:{args.ltype},topk:{args.topk},lr:{args.lr},batch-size:{args.batch_size},stop_cnt:{args.stop_cnt},epochs:{args.epochs}')
    print(f'global weight:{args.global_weight}')

    base_path = f'./output/temp/CKD/{args.dataset}/link/'
    name2id, id2name, features, node2neigh_list, _ = load_data(ltypes=[int(i) for i in args.ltype.strip().split(',')],
                                                               base_path=base_path,
                                                               use_features=True if args.attributed == 'True' else False)

    print(f'load data finish:{datetime.now()}')
    print('graph num:', len(node2neigh_list))
    print('node num:', len(name2id))

    target_nodes = np.array(list(id2name.keys()))
    if args.attributed != "True":
        features = np.random.randn(len(target_nodes), args.size).astype(np.float32)
    embeddings = torch.from_numpy(features).float().to(args.device)
    shuffle_embeddings = torch.from_numpy(shuffle(features)).to(args.device)

    dim = embeddings.shape[-1]

    adjs, sim_matrix_list = PPR(node2neigh_list)
    print('load adj finish', datetime.now())
    total_train_views = get_topk_neigh_multi(target_nodes, node2neigh_list, args.topk, adjs, sim_matrix_list)
    print(f'sample finish:{datetime.now()}')
    for node, status, view in total_train_views:
        for channel_data in view:
            channel_data[0] = torch.from_numpy(channel_data[0]).long().to(args.device)
            channel_data[1] = torch.from_numpy(channel_data[1]).float().to(args.device)
            # channel_data[0] = channel_data[0].long()
            data = embeddings[channel_data[0]]
            channel_data.append(data.reshape(1, data.shape[0], data.shape[1]))
            shuffle_data = shuffle_embeddings[channel_data[0]]

            channel_data.append(shuffle_data.reshape(1, shuffle_data.shape[0], shuffle_data.shape[1]))

    sample_train_views = [i for i in total_train_views if sum(i[1]) >= 1]
    print(f'context subgraph num:{len(sample_train_views)}')

    print(f'sample finish:{datetime.now()}')
    out_dim = args.dim
    model = CKDmodel(dim, out_dim, layers=args.layers)
    # model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCEWithLogitsLoss()

    stop_cnt = args.stop_cnt
    min_loss = 100000
    for epoch in range(args.epochs):
        if stop_cnt <= 0:
            break

        print(f'run epoch{epoch}')
        losses = []
        local_losses = []
        global_losses = []
        train_views = shuffle(sample_train_views)
        steps = (len(train_views) // args.batch_size) + (0 if len(train_views) % args.batch_size == 0 else 1)

        # get global emb
        global_graph_emb_list = []
        neg_global_graph_emb_list = []

        for channel in range(len(node2neigh_list)):
            train_features = torch.cat([i[2][channel][2] for i in total_train_views], dim=0)
            neg_features = torch.cat([i[2][channel][3] for i in total_train_views], dim=0)
            train_adj = torch.cat([i[2][channel][1] for i in total_train_views], dim=0)
            emb, graph_emb = model(train_features, train_adj)
            neg_emb, neg_graph_emb = model(neg_features, train_adj)
            index = torch.Tensor([0]).long().cpu()
            emb = emb.index_select(dim=1, index=index).squeeze().cpu()
            global_emb = torch.mean(emb, dim=0).detach().cpu()

            neg_emb = neg_emb.index_select(dim=1, index=index).squeeze()
            global_neg_emb = torch.mean(neg_emb, dim=0).detach()

            global_graph_emb_list.append(global_emb)
            neg_global_graph_emb_list.append(global_neg_emb)

        for step in tqdm(range(steps)):
            start = step * args.batch_size
            end = min((step + 1) * args.batch_size, len(train_views))
            if end - start <= 1:
                continue
            step_train_views = train_views[start:end]

            emb_list = []
            graph_emb_list = []
            for channel in range(len(node2neigh_list)):
                train_features = torch.cat([i[2][channel][2] for i in step_train_views], dim=0)
                train_adj = torch.cat([i[2][channel][1] for i in step_train_views], dim=0)
                emb, graph_emb = model(train_features, train_adj)
                emb_list.append(emb)
                graph_emb_list.append(graph_emb)

            local_loss = score(criterion, emb_list, graph_emb_list, [i[1] for i in step_train_views])
            global_loss = global_score(criterion, emb_list, global_graph_emb_list, neg_global_graph_emb_list,
                                       [i[1] for i in step_train_views])
            loss = local_loss + global_loss * args.global_weight
            losses.append(loss.item())
            local_losses.append(local_loss.item())
            global_losses.append(global_loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = np.mean(losses)
        print(f'epoch:{epoch},loss:{np.mean(losses)},{np.mean(local_losses)},{np.mean(global_losses)}')
        print(f'min_loss:{min_loss},epoch_loss:{epoch_loss}', epoch_loss < min_loss)

        if args.attributed == "True":
            emb_list = []
            eval_size = args.batch_size
            eval_steps = (len(total_train_views) // args.batch_size) + (
                0 if len(total_train_views) % args.batch_size == 0 else 1)
            for channel in range(len(node2neigh_list)):
                temp_emb_list = []
                for eval_step in range(eval_steps):
                    start = eval_step * eval_size
                    end = min((eval_step + 1) * eval_size, len(total_train_views))
                    step_eval_views = total_train_views[start:end]
                    train_features = torch.cat([i[2][channel][2] for i in step_eval_views], dim=0)
                    train_adj = torch.cat([i[2][channel][1] for i in step_eval_views], dim=0)
                    emb, graph_emb = model(train_features, train_adj)
                    index = torch.Tensor([0]).long().cuda()
                    emb = emb.index_select(dim=1, index=index).squeeze(dim=1)
                    emb = emb.cpu().detach().numpy()
                    temp_emb_list.append(emb)
                emb = np.concatenate(temp_emb_list, axis=0)
                emb_list.append(emb)
        else:
            emb_list = []
            for channel in range(len(node2neigh_list)):
                train_features = torch.cat([i[2][channel][2] for i in total_train_views], dim=0)
                train_adj = torch.cat([i[2][channel][1] for i in total_train_views], dim=0)
                emb, graph_emb = model(train_features, train_adj)
                index = torch.Tensor([0]).long().cpu()
                emb = emb.index_select(dim=1, index=index).squeeze().cpu()
                emb_list.append(emb)
        # epoch_auc=load_link_test_data(args.dataset,emb_list,name2id,)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            stop_cnt = args.stop_cnt
            output(args, emb_list[:1], id2name, need_handle=True)
            print(
                f'--------------------------------------------------------------------------------------------------------')
            """
            print(
                f'--------------------------------------------------------------------------------------------------------')
            print(
                f'-------------------------------------------save auc{epoch_auc}-------------------------------------------')
            print(
                f'--------------------------------------------------------------------------------------------------------')
            """

    # 调用函数删除指定文件夹下的所有文件
    folder_to_clear = f'./output/temp/CKD/{args.dataset}/link'
    delete_files_in_folder(folder_to_clear)


# if __name__ == '__main__':
#     main()
