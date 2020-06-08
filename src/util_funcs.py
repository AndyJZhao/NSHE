"""
data_loader and some basic functions for networks.
"""
import scipy.sparse as sp
import numpy as np
import psutil
import json
import os
import torch
import pandas as pd
import sys
import logging
from tqdm import tqdm as tqdm_base


def shell_init(gpu_id=1, server='S5'):
    '''
    Ignore wanrnings
    Set paths
    :param gpu_id:
    :return:
    '''
    import warnings
    warnings.filterwarnings("ignore")
    SOURCE_PATH = '/home/zja/PyProject/NSHE_handout/src/'
    os.chdir(SOURCE_PATH)
    if server != 'Xy':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    python_command = 'python' if server != 'Xy' else '/home/chopin/zja/anaconda3/bin/python'
    return python_command


def tqdm_fixed(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def load_features(in_path, dataset='dblp', node_types=None):
    true_feaure = {}
    for t in node_types:
        fname = in_path + dataset + "_" + t + "_feat.csv"
        # f_name = data_path + dataset + "_" + t + "_feat.csv!!!!!!!!"  # if use no real feature
        if os.path.exists(fname):
            true_feaure[t] = np.genfromtxt(fname, delimiter=',')
            true_feaure[t] = normalize_features(true_feaure[t])
        else:
            true_feaure[t] = None
    return true_feaure


def load_nodes(dataset):
    # os.chdir('/home/zja/PyProject/NEviaSchema/src/')
    # print('Current path:{}'.format(os.getcwd()))
    in_path = '../data/' + dataset + '/node2id.txt'
    with open(in_path) as node_file:
        nodes = node_file.readlines()

    node_cnt = {}
    type_info = {}
    node_id = {}
    node2id = {}
    id2node = {}
    for i, line in enumerate(nodes):
        token = line.strip('\n').split('\t')
        if i == 0:
            node_cnt_all = int(token[0])
        elif i == 1:
            cur_type = token[0][0]
            type_info[cur_type] = {}
            type_info[cur_type]['start_id'] = int(token[1])
            node_id[cur_type] = {}
            node_id[cur_type][token[0]] = int(token[1])
        else:
            if cur_type != token[0][0]:  # change type
                cur_type = token[0][0]
                type_info[cur_type] = {}
                type_info[cur_type]['start_id'] = int(token[1])
                node_id[cur_type] = {}
                node_id[cur_type][token[0]] = int(token[1])
            else:
                node_id[cur_type][token[0]] = int(token[1])

        if i != 0:
            node2id[token[0]] = int(token[1])
            id2node[int(token[1])] = token[0]

    t_info = {}
    for t in node_id.keys():
        node_cnt[t] = len(node_id[t])
        type_info[t]['end_id'] = type_info[t]['start_id'] + node_cnt[t]
        t_info[t] = {}
        t_info[t]['cnt'] = node_cnt[t]
        t_info[t]['ind'] = range(type_info[t]['start_id'], type_info[t]['end_id'])
    return node_id, t_info, node2id, id2node


def load_relations(in_path, dataset="dblp", node_cnt=37682):
    # ============ Generate adj_mx ============
    with open("{}/relations.txt".format(in_path, dataset)) as edge_file:
        edge_lines = edge_file.readlines()

    row, col = [], []
    for line in edge_lines:
        token = line.strip('\n').split('\t')
        row.append(int(token[0]))
        col.append(int(token[1]))
    adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(node_cnt, node_cnt))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    _ = row.copy()
    row = row + col
    col = col + _
    edge = {"r": row, "c": col}
    # Normalization
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj = torch.FloatTensor(np.array(adj.todense())) #Very large 5500mb for dblp adj
    adj_torch = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_torch, edge, len(row), adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def judge_type(t_info, node_id):
    for t in t_info.keys():
        if node_id in t_info[t]['ind']:
            return t
        else:
            continue


# Disable
def block_logs():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


# Restore
def enable_logs():
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def get_vis_port(server_name):
    if server_name == 'S3':
        return 49927
    elif server_name == 'S4':
        return 8427


def save_dict_json(d, fname):
    with open(fname, 'w') as file:
        json.dump(d, file)


def load_dict_json(fname):
    with open(fname, 'r') as file:
        d = json.load(file)
    return d


def print_memory_cost(input=None):
    """
    input: d/variable
    """
    if input is None:
        info = psutil.virtual_memory()
        print('Memory Cost：{:.2f} Gb'.format(
            float(psutil.Process(os.getpid()).memory_info().rss)
            / 1073733258))
        # print(u'总内存：', info.total)
        # print(u'内存占比：', info.percent)
        # print(u'cpu个数：', psutil.cpu_count())
    elif isinstance(input, dict):
        # print d
        for key in input:
            print('<{}> memory cost：'.format(key), end=' ')
            print_memory_cost(input[key])
    else:
        # print single variable
        print('{} byte'.format(sys.getsizeof(input)))


def calc_mean(train_on_gpu, result_list):
    if train_on_gpu:
        return np.array([r.data.cpu().numpy() for r in result_list]).mean()
    else:
        return np.array([r.data.numpy() for r in result_list]).mean()


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)