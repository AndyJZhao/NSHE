# coding:utf-8
# author: Andy Zhao
import pickle
from util_funcs import *
from deepwalk import gen_deep_walk_feature
from sampler import gen_neg_edges, gen_ns_instances

sys.path.append("..")


class HIN(object):
    """
    Stores the information of HIN
    """

    def __init__(self, hp):
        self.seed = hp.seed
        self.dataset = hp.dataset
        #
        self.node_id, self.t_info, self.node_cnt_all, self.edge_cnt = None, None, None, None
        self.adj, self.adf2, self.dw_features, self.edge = None, None, None, None
        self.true_feature, self.feature = {}, {}
        self.ns_instances, self.ns_label = None, None
        self.node_types = None
        self.optimizer = None
        self.ns_neg_rate = hp.ns_neg_rate
        self.seed_set = []
        #
        np.random.seed(self.seed)
        if not isinstance(self.seed, (int,)) and self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
        else:
            torch.manual_seed(2019)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(2019)

        # ================== Gen network ==================
        print("Generating {} network...".format(self.dataset))
        self.node_id, self.t_info, self.node2id, self.id2node = load_nodes(self.dataset)
        print(self.t_info)
        self.node_types = self.node_id.keys()
        # ================== Gen adj ==================
        self.adj, self.edge, self.edge_cnt, self.adj2 = load_relations(hp.data_path, self.dataset,
                                                                       len(self.node2id))

        # ================== Gen feature ==================
        dw_feat_file = hp.data_path + "/dw_emb_features.npy"
        if hp.init_dw_emb:
            print("Generating DeepWalk embedding of {}".format(self.dataset))
            self.dw_features = np.array(gen_deep_walk_feature(self.adj2))
            # origin_dw_features = np.array(gen_deep_walk_feature(self.adj2))
            # self.dw_features = normalize(origin_dw_features, axis=1)
            np.save(dw_feat_file, np.asarray(self.dw_features, dtype=np.float32))

            print("DeepWalk embeddings for {} saved.".format(self.dataset))
        else:
            self.dw_features = np.load(dw_feat_file)
        # ==================Combining features==================
        self.true_feature = load_features(hp.data_path, self.dataset, self.node_types)
        for t in self.node_types:
            if self.true_feature[t] is None:
                self.feature[t] = torch.FloatTensor(self.dw_features[self.t_info[t]['ind']])
            else:
                self.feature[t] = \
                    torch.FloatTensor(np.concatenate((
                        self.dw_features[self.t_info[t]['ind']],
                        self.true_feature[t]), axis=1))

            if hp.train_on_gpu:
                self.feature[t] = self.feature[t].cuda()
        # ==================sampling==================
        # self.neg_edge = gen_neg_edges(self.adj, self.edge, hp.e_neg_rate)
        # self.ns_instances, self.ns_label = gen_ns_instances(self.adj, self.edge, self.t_info, self.special_type,
        #                                                     self.ns_neg_rate)
        # Generate sampling seed for each epoch
        for i in range(hp.epochs):
            self.seed_set.append(np.random.randint(1000))

        if hp.train_on_gpu:
            self.adj = self.adj.cuda()

    def get_epoch_samples(self, epoch, hp):
        """
        Renew ns_instances and neg_edges in every epoch:
        1. get the seed for current epoch
        2. find using seed
            Y: load the file
            N: sample again and save
        """

        # seed for current epoch
        epoch_seed = self.seed_set[epoch]
        np.random.seed(epoch_seed)

        def _get_neg_edge(epoch_seed):
            fname = '../res/{}/neg_edges/NE-rate={:.0f}_seed={}.dat'.format(
                self.dataset, hp.e_neg_rate, epoch_seed)
            if os.path.exists(fname):
                # load
                with open(fname, 'rb') as handle:
                    try:
                        epoch_data = pickle.load(handle)
                        self.neg_edge = epoch_data['neg_edge']
                    except EOFError:
                        os.remove(fname)
                        print(epoch_seed, fname)
            else:
                # sample
                self.neg_edge = gen_neg_edges(self.adj2, self.edge, hp.e_neg_rate)
                # save
                data_to_save = {'neg_edge': self.neg_edge}
                with open(fname, 'wb') as handle:
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def _get_ns_instance(epoch_seed):
            fname = '../res/{}/network_schema_instances/NS-rate={:.0f}_seed={}.dat'.format(
                self.dataset, hp.ns_neg_rate, epoch_seed)
            if os.path.exists(fname):
                # load
                with open(fname, 'rb') as handle:
                    try:
                        epoch_data = pickle.load(handle)
                    except EOFError:
                        print(epoch_seed, fname)
                self.ns_instances = epoch_data['ns_instances']
                self.ns_label = epoch_data['ns_label']
            else:

                f_type_adj = '../data/{}/relation2id.txt'.format(self.dataset)
                self.ns_instances, self.ns_label = gen_ns_instances(f_type_adj, self.adj2, self.edge, self.t_info,
                                                                    self.ns_neg_rate)
                # save
                data_to_save = {
                    'ns_instances': self.ns_instances,
                    'ns_label': self.ns_label}
                with open(fname, 'wb') as handle:
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print('epoch_seed={}'.format(epoch_seed))
        _get_neg_edge(epoch_seed)
        _get_ns_instance(epoch_seed)

        if hp.train_on_gpu:
            self.ns_label = self.ns_label.cuda()
        return
