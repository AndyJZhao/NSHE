class Hyperparams:
    '''Hyperparameters'''

    # + Parameters for all data #
    # ! Train
    vis_flag = False
    eval_freq = 1
    eval_flag = True
    train_on_gpu = True
    seed = 0
    # ! Data
    # init_dw_emb = True # The first time Only
    init_dw_emb = False # Otherwise False
    norm_emb_flag = True
    dw_feat_only = False
    # ! Model config
    cla_method = 'TypeSpecCla'
    cla_layers = 2
    ns_emb_mode = 'TypeSpecCla'
    conv_method = 'GCNx1'
    size = {'com_feat_dim': 128, 'emb_dim': 128}
    e_neg_rate, ns_neg_rate = 1, 4  # ACM best, DBLP best (2layer)
    opt_method = 'Adam'
    lr_decay = 0.0
    weight_decay = 0.0
    def __init__(self, dataset, task='cla'):

        self.dataset = dataset
        self.data_path = "../data/" + dataset + '/'
        # ! Task Specific Settings
        optim_alpha = {'acm': {'cla': 0.001, 'clu': 0.001},
                       'dblp': {'cla': 0.008, 'clu': 0.008},
                       'imdb': {'cla': 0.008, 'clu': 0.008}}
        optim_beta = {'acm': {'cla': 0.135, 'clu': 33.115},
                      'dblp': {'cla': 0.251, 'clu': 0.905},
                      'imdb': {'cla': 0.05, 'clu': 0.05}}
        optim_epoch = {'acm': {'cla': 1200, 'clu': 300},
                       'dblp': {'cla': 100, 'clu': 200},
                       'imdb': {'cla': 700, 'clu': 500}}
        self.alpha = optim_alpha[dataset][task]
        self.beta = optim_beta[dataset][task]
        self.epochs = optim_epoch[dataset][task]
        if dataset=='imdb':
            e_neg_rate, ns_neg_rate = 4, 4
        else:
            e_neg_rate, ns_neg_rate = 1, 4
