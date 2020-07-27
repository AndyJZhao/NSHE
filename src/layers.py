import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn

cuda = False
cuda = True


class GraphConvolution(Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, global_W=None):
        if len(adj._values()) == 0:
            if cuda:
                return torch.zeros(adj.shape[0], self.out_features).cuda()
            else:
                return torch.zeros(adj.shape[0], self.out_features)
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        if global_W is not None:
            support = torch.spmm(support, global_W)  # Ignore this!
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphAttentionConvolution, self).__init__()
        self.out_dim = out_features
        self.weights = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weights.data, gain=1.414)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        self.attention = Attention_InfLevel(out_features)

    def forward(self, input, adj, global_W=None):

        h = torch.spmm(input, self.weights)
        # GAT 中的 HW
        h_prime = self.attention(h, adj) + self.bias
        return h_prime


class Attention_InfLevel(nn.Module):
    def __init__(self, dim_features):
        super(Attention_InfLevel, self).__init__()
        self.dim_features = dim_features
        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        N = h.size()[0]
        e1 = torch.matmul(h, self.a1).repeat(1, N)
        e2 = torch.matmul(h, self.a2).repeat(1, N).t()
        e = e1 + e2
        e = self.leakyrelu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        # zero_vec = torch.zeros_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        del zero_vec
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # h' = alpha * h(hw)
        # print('Node attention vector:{}'.format(self.a1[:5]))
        return h_prime


class TypeLevelAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=50):
        super(TypeLevelAttention, self).__init__()
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = Parameter(torch.FloatTensor(hidden_dim, 1))
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        """
        :param inputs: N(num_ns_ins) * n_type * in_features
        对于每个network schema instance, 一共有n_type种邻居聚合而来的信息.
        :return:经过 type attention之后的ns_instance 表示.
        """
        # inputs:
        # output: 类型之间的 attention
        x = self.tanh(self.linear(inputs))
        # calculate local type attention score (attention score for every instance)
        U = torch.matmul(x, self.a).squeeze()
        # calculate global type attention score (average attenetion score for all the instance)
        global_score = torch.sum(U, dim=0) / U.shape[0]
        # N * n_type * 1  每个节点对不同type 的 attention score
        weights = F.softmax(global_score)  # 每个节点对不同type的attention值 公式 5 alpha^_tau
        outputs = torch.matmul(inputs.transpose(1, 2), weights)
        # print('Type Attention Vector [:5]={}\nWeight vector:{}, '.format(self.a[:5], weights))
        return outputs, weights
