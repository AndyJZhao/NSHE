import argparse
import warnings
from models import NSHE
from hyperparams import Hyperparams
from evaluation import Evaluation
import torch.nn.functional as F
import torch.optim as optim
import time
from util_funcs import *
from hin_loader import HIN

sys.path.append("..")
shell_init(gpu_id=2)


def cal_pairwise_ce_loss(node_emb, edge, neg_edge):
    # cross entropy loss from LINE
    # pos loss
    emb_u_i = node_emb[edge['r']]
    emb_u_j = node_emb[edge['c']]
    inner_product = torch.sum(emb_u_i * emb_u_j, dim=1)
    pos_loss = - torch.mean(F.logsigmoid(inner_product))
    # neg loss
    emb_u_i = node_emb[neg_edge['r']]
    emb_u_j = node_emb[neg_edge['c']]
    inner_product = torch.sum(emb_u_i * emb_u_j, dim=1)
    neg_loss = - torch.mean(F.logsigmoid(-1 * inner_product))
    loss = pos_loss + neg_loss
    return loss


def cal_cla_loss(predict, ns_label):
    BCE_loss = torch.nn.BCELoss()
    return BCE_loss(predict, ns_label)


def evaluate(node_emb, t_info):
    # Evaluation
    # Evaluate with random seed
    exp = Evaluation(hp.dataset)
    if hp.dataset == 'imdb':
        res = exp.evaluate_imdb(node_emb[:t_info['m']['cnt']])
    elif hp.dataset == 'dblp':
        res = exp.evaluate_dblp(node_emb)
    elif hp.dataset == 'acm':
        res = exp.evaluate_acm(node_emb)
    print_dict(res)


def run(model, g, optimizer):
    for epoch in range(hp.epochs):
        epoch_start_time = time.time()
        g.get_epoch_samples(epoch, hp)
        model.train()  # set paras to train ns_emb_mode
        optimizer.zero_grad()
        node_emb, predict = model(g.adj, g.feature, g.ns_instances)
        pairwise_loss = cal_pairwise_ce_loss(node_emb, g.edge, g.neg_edge)
        cla_loss = cal_cla_loss(predict, g.ns_label)
        # loss = cla_loss
        loss = pairwise_loss + cla_loss * hp.beta
        loss.backward()
        epoch_dict = {'Epoch': epoch, 'train_loss': loss.item(),
                      'pairwise_loss': pairwise_loss.item(),
                      'cla_loss': cla_loss.item(),
                      'time': time.time() - epoch_start_time}
        print_dict(epoch_dict, '\n')
        optimizer.step()
    return node_emb


def main(hp):
    g = HIN(hp)
    model = NSHE(g, hp)
    if hp.train_on_gpu:
        model.cuda()
    if hp.opt_method == "Adam" or hp.opt_method == "adam":
        optimizer = optim.Adam(model.parameters(), lr=hp.alpha)
    else:
        optimizer = optim.SGD(model.parameters(), lr=hp.alpha)

    node_emb = run(model, g, optimizer)  # 模型训练
    print("Train finished")
    model.eval()
    evaluate(node_emb, g.t_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default='dblp', help="dataset to train")
    parser.add_argument("--task", type=str, default='cla', help="task to train")
    args = parser.parse_args()
    hp = Hyperparams(args.dataset, args.task)
    warnings.filterwarnings('ignore')
    main(hp)
