import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet, GTH, get_act, get_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, norm, act, dropout=0.5):
        super(SimpleClassifier, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        layers = [
            norm_layer(nn.Linear(in_dim, hid_dim), dim=None),
            act_layer(),
            nn.Dropout(dropout, inplace=False),
            norm_layer(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class PaperClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim_1, hid_dim_2, out_dim, norm, act, dropout=0.5):
        super(PaperClassifier, self).__init__()

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


        self.gated_tanh_1 = GTH(in_dim=in_dim, out_dim=hid_dim_1, dropout=dropout, norm=norm, act=act)
        self.gated_tanh_2 = GTH(in_dim=in_dim, out_dim=hid_dim_2, dropout=dropout, norm=norm, act=act)

        self.linear_1 = norm_layer(nn.Linear(hid_dim_1, out_dim), dim=None)
        self.linear_2 = norm_layer(nn.Linear(hid_dim_2, out_dim), dim=None)

    def forward(self, x):
        v_1 = self.gated_tanh_1(x)
        v_2 = self.gated_tanh_2(x)

        v_1 = self.linear_1(v_1)
        v_2 = self.linear_2(v_2)

        logits = v_1 + v_2
        return logits
