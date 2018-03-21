from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


def get_norm(norm):
    no_norm = lambda x, dim: x
    if norm == 'weight':
        norm_layer = weight_norm
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    elif norm == 'layer':
        norm_layer = nn.LayerNorm
    elif norm == 'none':
        norm_layer = no_norm
    else:
        print("Invalid Normalization")
        raise Exception("Invalid Normalization")
    return norm_layer


def get_act(act):
    if act == 'ReLU':
        act_layer = nn.ReLU
    elif act == 'LeakyReLU':
        act_layer = nn.LeakyReLU
    elif act == 'PReLU':
        act_layer = nn.PReLU
    elif act == 'RReLU':
        act_layer = nn.RReLU
    elif act == 'ELU':
        act_layer = nn.ELU
    elif act == 'SELU':
        act_layer = nn.SELU
    elif act == 'Tanh':
        act_layer = nn.Tanh
    elif act == 'Hardtanh':
        act_layer = nn.Hardtanh
    elif act == 'Sigmoid':
        act_layer = nn.Sigmoid
    else:
        print("Invalid activation function")
        raise Exception("Invalid activation function")
    return act_layer



class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, dropout, norm, act):
        super(FCNet, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(norm_layer(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))
        layers.append(norm_layer(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(act_layer())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class GTH(nn.Module):
    """Simple class for Gated Tanh
    """
    def __init__(self, in_dim, out_dim, dropout, norm, act):
        super(GTH, self).__init__()

        self.nonlinear = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= act)
        self.gate = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= 'Sigmoid')

    def forward(self, x):
        x_proj = self.nonlinear(x)
        gate = self.gate(x)
        x_proj = x_proj*gate
        return x_proj


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
