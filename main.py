import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
from models import build_baseline, build_model_A1, build_model_A2, build_model_A3,build_model_AP,\
    build_model_APD, build_model_APx2, build_model_AP_PC, build_model_P_exact,build_model_P_mod,\
    build_model_A3x2, build_model_A2x2, build_model_A23P, build_model_A3x3, build_model_A2x3, build_model_A3S
from train import train
from plot import plot_charts
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1280) # they used 1024
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_L', type=float, default=0.1)
    parser.add_argument('--dropout_G', type=float, default=0.2)
    parser.add_argument('--dropout_W', type=float, default=0.4)
    parser.add_argument('--dropout_C', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='PReLU, ReLU, LeakyReLU, Tanh, Hardtanh, Sigmoid, RReLU, ELU, SELU')
    parser.add_argument('--norm', type=str, default='weight', help='weight, batch, layer, none')
    parser.add_argument('--model', type=str, default='A3x2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adamax', help='Adam, Adamax, Adadelta, RMSprop')
    parser.add_argument('--initializer', type=str, default='kaiming_normal')
    parser.add_argument('--seed', type=int, default=9731, help='random seed')
    args = parser.parse_args()
    return args


def weights_init_xn(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)
def weights_init_xu(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

# a=0.01 for Leaky RelU
def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)
def weights_init_ku(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data, a=0.01)



if __name__ == '__main__':
    args = parse_args()

    seed = 0
    if args.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(args.seed)
    else:
        seed = args.seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output = args.output + args.model + '_' + str(args.num_hid) + '_' + args.activation + '_' + args.optimizer +\
            '_D' + str(args.dropout) + '_DL' + str(args.dropout_L) + '_DG' + str(args.dropout_G) + '_DW' + str(args.dropout_W) \
            + '_DC' + str(args.dropout_C) + '_w' + str(args.weight_decay) + '_SD' + str(seed) \
            + '_initializer_' + args.initializer

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)

    if args.model == 'baseline':
        model = build_baseline(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A1':
        model = build_model_A1(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A2':
        model = build_model_A2(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A3':
        model = build_model_A3(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A3S':
        model = build_model_A3S(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'AP':
        model = build_model_AP(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'APD':
        model = build_model_APD(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'APx2':
        model = build_model_APx2(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A3x2':
        model = build_model_A3x2(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A2x2':
        model = build_model_A2x2(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A23P':
        model = build_model_A23P(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A2x3':
        model = build_model_A2x3(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'A3x3':
        model = build_model_A3x3(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'AP_PC':
        model = build_model_AP_PC(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    elif args.model == 'PAPER_Exact':
        model = build_model_P_exact(train_dset, num_hid = 512, dropout = 0, norm ='none', activation = 'Tanh')
    elif args.model == 'PAPER_mod':
        model = build_model_P_mod(train_dset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, dropL=args.dropout_L, dropG=args.dropout_G,\
                               dropW=args.dropout_W, dropC=args.dropout_C)
    else:
        print("Invalid Model")
        sys.exit(0)

    model = model.cuda()

    if args.initializer == 'xavier_normal':
        model.apply(weights_init_xn)
    elif args.initializer == 'xavier_uniform':
        model.apply(weights_init_xu)
    elif args.initializer == 'kaiming_normal':
        model.apply(weights_init_kn)
    elif args.initializer == 'kaiming_uniform':
        model.apply(weights_init_ku)

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    batch_size = args.batch_size
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_dset = VQAFeatureDataset('val', dictionary)
    eval_loader  = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)

    train(model, train_loader, eval_loader, args.epochs, output, args.optimizer, args.weight_decay)
    plot_charts(output)

    """
    if args.eval:
        test(args)
    """
