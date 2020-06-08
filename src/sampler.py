import random
from util_funcs import *


class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, probs):
        self.n = len(probs)
        self.U = np.array(probs) * self.n
        self.K = [i for i in range(len(probs))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


def gen_neg_edges(adj, edge, neg_rate):
    node_degrees = np.squeeze(np.asarray(np.sum(adj, axis=0)))
    # node_degrees = torch.sparse.sum(adj, dim=0)
    # node_degrees = node_degrees.to_dense().cpu().numpy()
    node_weights = np.power(node_degrees, 0.75)
    node_sampler = AliasSampling(probs=node_weights / np.sum(node_weights))
    row, col = [], []
    # for i in tqdm_fixed(range(len(edge['r'])), desc='NegEdge-sampling'):
    for i in range(len(edge['r'])):
        for _ in range(neg_rate):
            while True:
                neg_node = node_sampler.sampling()
                if adj[edge['r'][i], neg_node] <= 1e-4:
                    break
            row.append(edge['r'][i])
            col.append(neg_node)
    neg_edges = {'r': row, 'c': col}
    return neg_edges


def gen_ns_instances(f_type_adj, adj, edge, t_info, ns_negrate=1):
    def _gen_type_adj():
        # read schema adjs
        type_adj = {}
        with open(f_type_adj, 'r') as r2i_file:
            lines = r2i_file.readlines()
        for t in types:
            type_adj[t] = set()
        for line in lines:
            tokens = line.strip().split('\t')
            if tokens[0][0] not in types:
                continue
            lr_type, ri_type = tokens[0][0], tokens[0][1]
            type_adj[lr_type].add(ri_type)
            type_adj[ri_type].add(lr_type)
        return type_adj

    def _get_current_sample_types(prev_sampled_types, pos_ns):
        current_sample_types = []
        for t in prev_sampled_types:
            for neighbor_t in type_adj[t]:
                if neighbor_t not in pos_ns.keys():
                    current_sample_types.append(neighbor_t)
        return current_sample_types

    def _get_type_specific_node_samplers():
        # node samplers for type
        node_degrees = np.squeeze(np.asarray(np.sum(adj, axis=0)))
        node_weights = np.power(node_degrees, 0.75)
        node_sampler = {}
        for t in types:
            t_weights = node_weights[t_info[t]['ind']]
            node_sampler[t] = AliasSampling(probs=t_weights / np.sum(t_weights))
        return node_sampler

    def _sample_pos_ns(i, target_t):
        id = t_info[target_t]['ind'].start + i  # id of the node
        pos_ns = {'target_type': target_t, target_t: id}
        prev_sampled_types = set([target_t])
        #
        current_sample_types = _get_current_sample_types(prev_sampled_types, pos_ns)
        while len(current_sample_types) != 0:
            # find neighbors of previous sampled nodes
            prev_neighbors = []
            for _ in prev_sampled_types:
                prev_node = pos_ns[_]
                prev_node_neighbors = col[row == prev_node]
                for _id in prev_node_neighbors: prev_neighbors.append(_id)
            prev_neighbors = np.array(prev_neighbors)
            if len(prev_neighbors) == 0:  # if there is no neighbor to select
                print('Node {} is a isolated point!!'.format(id))
                return None
            for t in current_sample_types:
                # find neighbors of type t
                t_neighbors = prev_neighbors[[i in t_info[t]['ind'] for i in prev_neighbors]]
                # random select one as postive ns_instance
                if len(t_neighbors) == 0:  # if there is no neighbor to select
                    print('Node {} has no {} type point!!'.format(id, t))
                    return None
                if len(t_neighbors) == 1:  # if there is only one selection
                    r = 0
                elif len(t_neighbors) >= 1:  # if there is only one selection
                    r = random.randint(0, len(t_neighbors) - 1)
                pos_ns[t] = t_neighbors[r]
            prev_sampled_types = current_sample_types
            current_sample_types = _get_current_sample_types(prev_sampled_types, pos_ns)
        pos_ns['labels'] = 1
        return pos_ns

    types = list(t_info.keys())
    type_adj = _gen_type_adj()
    row = np.array(edge['r'])
    col = np.array(edge['c'])
    node_sampler = _get_type_specific_node_samplers()
    ns_ins_list = []
    for target_t in types:
        # Sample using target_t
        for i in range(t_info[target_t]['cnt']):
            # ================== pos ns instnace sampling ==================
            pos_ns = _sample_pos_ns(i, target_t)
            if pos_ns is not None:
                ns_ins_list.append(pos_ns)
            else:
                continue
            # ================== neg ns instnace sampling ==================
            for _ in range(ns_negrate):
                neg_ns = pos_ns.copy()
                neg_node = node_sampler[target_t].sampling() + t_info[target_t]['ind'].start
                # replace type in schema instance
                neg_ns[target_t] = neg_node
                neg_ns['labels'] = 0  # BCE_loss, negative samples label = 0
                ns_ins_list.append(neg_ns)
    # ================== df generation ==================
    ns_pd = pd.DataFrame.from_records(ns_ins_list)
    col_names = types.copy()
    for _ in ['target_type', 'labels']: col_names.append(_)
    ns_pd = ns_pd[col_names].reset_index(drop=True)
    ns_pd = ns_pd.sample(frac=1).reset_index(drop=True)
    labels = torch.FloatTensor(np.array(ns_pd['labels']))
    # print(ns_pd.describe())
    return ns_pd, labels
