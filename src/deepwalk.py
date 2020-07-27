#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    deepwalk.py
  Author:       locke
  Date created: 2018/5/8 上午10:03

  Modified by Andy Zhao 2019/11/2
"""

import time
import random
import numpy as np
from gensim.models import Word2Vec


random.seed(2018)
np.random.seed(2018)


def gen_deep_walk_feature(A, number_walks=10, alpha=0, walk_length=80, window=10, workers=2, size=128):#,row, col,
    row,col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build adj_mat")
    t1 = time.time()
    G = {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = []  # 存放上下文的 list,每一个节点一个上下文(随机游走序列)
    for cnt in range(number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  # 对每个节点找到他的游走序列.
            while len(path) < walk_length:
                cur = path[-1]  # 每次从序列的尾记录当前游走位置.
                if len(G[cur]) > 0:
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur]))  # 如果有邻居,邻接矩阵里随便选一个
                    else:
                        path.append(path[0])  # 如果没有,就是自己
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print("cost: {}s".format(t2 - t1))
    print("train...")
    model = Word2Vec(corpus,
                     size=size,  # emb_size
                     window=window,
                     min_count=0,
                     sg=1,  # skip gram
                     hs=1,  # hierarchical softmax
                     workers=workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:  # word2vec 的输出以字典的形式存在.wv 里
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    return output
