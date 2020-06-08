# coding: utf-8
# author: lu yf
# create date: 2017/12/29

from __future__ import division

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
import warnings
from util_funcs import *

warnings.filterwarnings('ignore')


class Evaluation:
    def __init__(self, dataset='imdb'):
        self.dataset = dataset
        if dataset == 'imdb':
            # read labels
            self.labels = []
            with open('../data/imdb/imdb_m_label.txt', 'r') as l_file:
                l_lines = l_file.readlines()
            for line in l_lines:
                self.labels.append(int(line.strip('\n')))

    # self.labels[a] and self.labels[p]

    def avg_list(self, list):
        return sum(list) / len(list)

    def load_emb(self, emb_name):
        """
        load embeddings
        :param emb_name:
        :return:
        """
        with open(emb_name, 'r') as emb_file:
            emb_dict = json.load(emb_file)
        return emb_dict

    def evaluate_imdb(self, emb_tensor):
        k = 3
        x = emb_tensor.tolist()

        nmi = self.kmeans_nmi(x, self.labels, k)
        mi_f1, ma_f1 = self.classification(x, self.labels)
        nmi_ = '{:.4f}'.format(nmi)
        mi_f1_ = '{:.4f}'.format(mi_f1)
        ma_f1_ = '{:.4f}'.format(ma_f1)
        return {'nmi': nmi_, 'mi_f1': mi_f1_, 'ma_f1': ma_f1_}

    def evaluate_acm(self, emb_tensor):
        def _load_labels():
            x_a, l_a = [], []
            # Read Paper Labels
            with open('../data/' + self.dataset + '/p_label.txt', 'r') as l_file:
                l_lines = l_file.readlines()
            for line in l_lines:
                tokens = line.strip().split('\t')
                x_a.append(int(tokens[0]))
                l_a.append(int(tokens[1]))
            return x_a, l_a

        def _eval(emb_tensor, k, return_string=False):
            x = emb_tensor[eval_ids, :].tolist()
            mi_f1, ma_f1 = self.classification(x, labels)
            nmi = self.kmeans_nmi(x, labels, k)
            if return_string:
                return '{:.4f}'.format(nmi), \
                       '{:.4f}'.format(mi_f1), '{:.4f}'.format(ma_f1)
            else:
                return nmi, mi_f1, ma_f1

        k = 3
        _, t_info, node2id, _ = load_nodes(self.dataset)
        eval_ids, labels = _load_labels()
        nmi, mi_f1, ma_f1 = _eval(emb_tensor, k, True)

        return {'nmi': nmi, 'mi_f1': mi_f1, 'ma_f1': ma_f1}

    def evaluate_yelp(self, emb_tensor):
        def _load_labels():
            x_a, l_a = [], []
            # Read Business Labels
            with open('../data/' + self.dataset + '/bus_label.txt', 'r') as l_file:
                l_lines = l_file.readlines()
            for line in l_lines:
                tokens = line.strip().split('\t')
                x_a.append(int(tokens[0]))
                l_a.append(int(tokens[1]))
            return x_a, l_a

        def _eval(emb_tensor, k, return_string=False):
            x = emb_tensor[eval_ids, :].tolist()
            mi_f1, ma_f1 = self.classification(x, labels)
            nmi = self.kmeans_nmi(x, labels, k)
            if return_string:
                return '{:.4f}'.format(nmi), \
                       '{:.4f}'.format(mi_f1), '{:.4f}'.format(ma_f1)
            else:
                return nmi, mi_f1, ma_f1
        k = 3
        _, t_info, node2id, _ = load_nodes(self.dataset)
        eval_ids, labels = _load_labels()
        nmi, mi_f1, ma_f1 = _eval(emb_tensor, k, True)
        return {'nmi': nmi, 'mi_f1': mi_f1, 'ma_f1': ma_f1}

    def evaluate_dblp(self, emb_tensor):

        def _load_labels():
            x_a, x_p, l_a, l_p = [], [], [], []
            # Read Author Labels
            with open('../data/' + self.dataset + '/author_label.txt', 'r') as l_file:
                l_lines = l_file.readlines()
            for line in l_lines:
                tokens = line.strip().split('\t')
                # x_a.append(int(tokens[0]))
                x_a.append(node2id['a' + tokens[0]])
                l_a.append(int(tokens[1]))
            # Read Paper Labels
            with open('../data/' + self.dataset + '/paper_label.txt', 'r') as l_file:
                l_lines = l_file.readlines()
            for line in l_lines:
                tokens = line.strip().split('\t')
                # x_p.append(int(tokens[0]))
                x_p.append(node2id['p' + tokens[0]])
                l_p.append(int(tokens[1]))

            eval_ids = {'a': x_a, 'p': x_p}
            labels = {'a': l_a, 'p': l_p}
            return eval_ids, labels

        def _eval(emb_tensor, type, k):
            # node name in
            x = emb_tensor[eval_ids[type], :].tolist()
            nmi = self.kmeans_nmi(x, labels[type], k)
            mi_f1, ma_f1 = self.classification(x, labels[type])
            return nmi, mi_f1, ma_f1

        def _eval_return_string(emb_tensor, type, k):
            # node name in
            x = emb_tensor[eval_ids[type], :].tolist()
            # emb = emb_tensor[eval_ids[type], :]
            # np.save('../visualization/author_emb', emb.detach().cpu().numpy())
            nmi = self.kmeans_nmi(x, labels[type], k)
            mi_f1, ma_f1 = self.classification(x, labels[type])
            # np.save('../visualization/p_label', labels[type])
            return '{:.4f}'.format(nmi), '{:.4f}'.format(mi_f1), '{:.4f}'.format(ma_f1)

        k = 4
        _, t_info, node2id, _ = load_nodes(self.dataset)
        eval_ids, labels = _load_labels()
        a_nmi_list, p_nmi_list, a_mif1_list, a_maf1_list, p_mif1_list, p_maf1_list = [], [], [], [], [], []

        p_nmi, p_mi_f1, p_ma_f1 = _eval_return_string(emb_tensor, 'p', k)
        a_nmi, a_mi_f1, a_ma_f1 = _eval_return_string(emb_tensor, 'a', k)

        return {'p_nmi': p_nmi, 'p_mi_f1': p_mi_f1, 'p_ma_f1': p_ma_f1, 'a_nmi': a_nmi, 'a_mi_f1': a_mi_f1,
                'a_ma_f1': a_ma_f1}

    def kmeans_nmi(self, x, y, k):
        km = KMeans(n_clusters=k)
        km.fit(x, y)
        y_pre = km.predict(x)

        nmi = normalized_mutual_info_score(y, y_pre)
        # print('NMI: {}'.format(nmi))
        return nmi

    def classification(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        mi_f1 = f1_score(y_valid, y_valid_pred, average='micro')
        ma_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        return mi_f1, ma_f1
