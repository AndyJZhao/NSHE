# coding:utf-8
# author: Andy Zhao
# create date: 2018/2/7
from layers import *


# a little trick for layer lists
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class NS_MLP_Classifier(nn.Module):
    def __init__(self, in_feat, hidden_dim=[16]):
        super(NS_MLP_Classifier, self).__init__()
        self.hidden_layer = nn.Linear(in_feat, hidden_dim[0])
        self.output_layer = nn.Linear(hidden_dim[-1], 1)
        return

    def forward(self, input):
        ns_x = F.relu(self.hidden_layer(input))
        ns_y = self.output_layer(ns_x)
        ns_y = F.sigmoid(ns_y).flatten()
        return ns_y


class NSHE(nn.Module):

    def __init__(self, g, hp):
        super(NSHE, self).__init__()
        self.conv_method = hp.conv_method
        self.cla_layers = hp.cla_layers
        self.ns_emb_mode = hp.ns_emb_mode
        self.cla_method = hp.cla_method
        self.norm_emb = hp.norm_emb_flag
        self.types = g.node_types
        size = hp.size
        self.t_info = g.t_info
        for t in self.types:
            self.add_module('encoder_' + t, nn.Linear(g.feature[t].shape[1], size['com_feat_dim']))
        self.encoder = AttrProxy(self, 'encoder_')
        self.non_linear = nn.ReLU()
        self.context_dim = int(size['emb_dim'] / (len(self.types) - 1))
        # * ================== Neighborhood Agg==================
        emb_dim = size['emb_dim']
        if self.conv_method[:3] == 'GAT':
            self.neig_aggregator = GraphAttentionConvolution(size['com_feat_dim'], size['emb_dim'])
            if self.conv_method[-1] == '2':
                emb_dim = int(size['emb_dim'] / 2)
                self.neig_aggregator_2 = GraphAttentionConvolution(size['emb_dim'], emb_dim)
        elif self.conv_method[:3] == 'GCN':
            self.neig_aggregator = GraphConvolution(size['com_feat_dim'], size['emb_dim'])
            if self.conv_method[-1] == '2':
                emb_dim = int(size['emb_dim'] / 2)
                self.neig_aggregator_2 = GraphConvolution(size['emb_dim'], emb_dim)
        # * ================== NSI Embedding Gen=================
        if self.cla_method == 'TypeSpecCla':
            for t in self.types:
                self.add_module('nsi_encoder' + t, nn.Linear(emb_dim, self.context_dim))
            self.nsi_encoder = AttrProxy(self, 'nsi_encoder')
        # * ================== NSI Classification================
        if self.cla_method == '2layer':
            if self.ns_emb_mode == 'TypeLvAtt':
                self.ns_classifier = NS_MLP_Classifier(
                    emb_dim, [int(emb_dim / 2)])
            elif self.ns_emb_mode == 'Concat':
                self.ns_classifier = NS_MLP_Classifier(len(g.t_info) * emb_dim, emb_dim)
        elif self.cla_method == 'TypeSpecCla':
            for t in self.types:
                if self.cla_layers == 1:
                    self.add_module('ns_cla_' + t, nn.Linear(emb_dim + self.context_dim * (len(self.types) - 1), 1))
                else:
                    self.add_module('ns_cla_' + t,
                                    NS_MLP_Classifier(emb_dim + self.context_dim * (len(self.types) - 1), [16]))
            self.ns_classifier = AttrProxy(self, 'ns_cla_')
        print(self)

    def forward(self, adj, features, nsi_list):
        # * =============== Encode heterogeneous feature ================
        #
        encoded = torch.cat([self.non_linear(self.encoder[t](features[t])) for t in self.types])
        # * =============== Node Embedding Generation ===================
        com_emb = self.neig_aggregator(encoded, adj)
        if self.conv_method[-1] == '2':
            com_emb = self.neig_aggregator_2(com_emb, adj)
        if self.norm_emb:
            # Independently normalize each dimension
            com_emb = F.normalize(com_emb, p=2, dim=1)
        #
        # * =============== NSI Embedding Classification ================
        if self.cla_method == 'TypeSpecCla':
            # Type Specific Encoder
            context_emb = torch.FloatTensor(torch.zeros(com_emb.shape[0], self.context_dim)).cuda()
            ns_y = torch.FloatTensor([-1] * len(nsi_list)).cuda()
            for t in self.types:
                context_emb[self.t_info[t]['ind']] = self.non_linear(
                    self.nsi_encoder[t](com_emb[self.t_info[t]['ind']]))  # Get NSI Embedding
            # Type Specific Classifier
            for t in self.types:
                nsi_index = nsi_list[nsi_list['target_type'] == t].index.tolist()
                # node_list generation
                nsi_node_list = {}
                for ct in self.types:
                    nsi_node_list[ct] = list(nsi_list[ct][nsi_index])
                # Context embedding generation
                context_combined = torch.cat(
                    [context_emb[nsi_node_list[ct]] for ct in self.types if ct != t], dim=1)
                # NSI Embedding generation: concat context and target embs
                ns_xt = torch.cat([context_combined, com_emb[nsi_node_list[t]]], dim=1)
                if self.cla_layers == 1:
                    ns_y[nsi_index] = F.sigmoid(self.ns_classifier[t](ns_xt)).flatten()
                else:
                    ns_y[nsi_index] = self.ns_classifier[t](ns_xt)
        else:
            ns_ins_emb = [com_emb[nsi_list[t]] for t in self.types]
            if self.ns_emb_mode == 'Concat':
                ns_x = torch.cat(ns_ins_emb, dim=1)
            elif self.ns_emb_mode == 'TypeLvAtt':
                # gen ns_embedding via type attention
                ns_ins_emb = torch.stack(ns_ins_emb, dim=1)
                ns_x, weight = self.type_attention_layer(ns_ins_emb)
            # predict ns_instance labels
            if self.cla_method == '2layer':
                ns_y = self.ns_classifier(ns_x)
        return com_emb, ns_y
