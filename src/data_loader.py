import scipy.sparse as sp
import numpy as np
import os
import torch


def load_features(in_path, dataset='dblp', node_types=None):
    true_feaure = {}
    for t in node_types:
        fname = in_path + dataset + t + "_feat.txt"
        if os.path.exists(fname):
            true_feaure[t] = load_feat(fname)
        else:
            true_feaure[t] = None
    return true_feaure


def load_feat(fname):
    return None


def load_nodes(in_path, dataset="dblp"):
    # Where to use ?
    print('Loading {} dataset...'.format(dataset))
    # ============ Generate adj_mx ============
    with open("{}/o_node2id.txt".format(in_path, dataset)) as node_file:
        nodes = node_file.readlines()

    node_cnt = {}
    node_base_id = {}
    node_id = {}
    for i, line in enumerate(nodes):
        token = line.strip('\n').split('\t')
        if i == 0:
            node_cnt_all = int(token[0])
        elif i == 1:
            cur_type = token[0][0]
            node_base_id[cur_type] = int(token[1])
            node_id[cur_type] = {}
            node_id[cur_type][token[0]] = int(token[1])
        else:
            if cur_type != token[0][0]:  # change type
                cur_type = token[0][0]
                node_base_id[cur_type] = int(token[1])
                node_id[cur_type] = {}
                node_id[cur_type][token[0]] = int(token[1])
            else:
                node_id[cur_type][token[0]] = int(token[1])

    for t in node_id.keys():
        node_cnt[t] = len(node_id[t])
    return node_cnt, node_id, node_base_id, node_cnt_all


def load_relations(in_path, dataset="dblp", node_cnt=37682):
    print('Loading {} dataset...'.format(dataset))
    # ============ Generate adj_mx ============
    with open("{}/relations.txt".format(in_path, dataset)) as edge_file:
        edges = edge_file.readlines()

    row, col = [], []
    for line in edges:
        token = line.strip('\n').split('\t')
        row.append(token[0])
        col.append(token[1])
    adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(node_cnt, node_cnt))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj = torch.FloatTensor(np.array(adj.todense())) Very large
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


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


def read_graph(graph_filename):
    # p -> a : 0
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    # graph_filename = '../data/yelp/yelp_triple.dat'

    relations = set()
    nodes = set()
    graph = {}

    with open(graph_filename) as infile:
        for line in infile.readlines():
            source_node, target_node, relation = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)
            relation = int(relation)

            nodes.add(source_node)
            nodes.add(target_node)
            relations.add(relation)

            if source_node not in graph:
                graph[source_node] = {}

            if relation not in graph[source_node]:
                graph[source_node][relation] = []

            graph[source_node][relation].append(target_node)

    n_node = len(nodes)
    # print relations
    return n_node, len(relations), graph


if __name__ == "__main__":
    dataset = 'yelp'
    source_path = '/home/zja/PyProject/NEviaSchema/src/'
    os.chdir(source_path)
    graph_filename = '../data/yelp/yelp_triple.dat'
    read_graph(graph_filename)
