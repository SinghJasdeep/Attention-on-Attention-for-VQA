import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet, GTH, get_norm


# Default concat, 1 layer, output layer
class Att_0(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_0, self).__init__()
        norm_layer = get_norm(norm)
        self.nonlinear = FCNet([v_dim + q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


# concat, 2 layer, output layer
class Att_1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_1, self).__init__()
        norm_layer = get_norm(norm)
        self.nonlinear = FCNet([v_dim + q_dim, num_hid, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


# 1 layer seperate, element-wise *, output layer
class Att_2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_2, self).__init__()
        norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.q_proj = FCNet([q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        logits = self.linear(joint_repr)
        return logits


# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_3, self).__init__()
        norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.q_proj = FCNet([q_dim, num_hid], dropout= dropout, norm= norm, act= act)
        self.nonlinear = FCNet([num_hid, num_hid], dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1) # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr)
        return logits

# 1 layer seperate, element-wise *, 1 layer seperate, output layer
class Att_3S(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_3S, self).__init__()
        norm_layer = get_norm(norm)
        self.v_proj = FCNet([v_dim, num_hid], dropout=dropout, norm=norm, act=act)
        self.q_proj = FCNet([q_dim, num_hid], dropout=dropout, norm=norm, act=act)
        self.nonlinear = FCNet([num_hid, num_hid], dropout=dropout, norm=norm, act=act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.sigmoid(logits)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)  # [batch, k, num_hid]
        joint_repr = v_proj * q_proj
        joint_repr = self.nonlinear(joint_repr)
        logits = self.linear(joint_repr)
        return logits


# concat w/ 2 layer seperate, element-wise *, output layer
class Att_PD(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_PD, self).__init__()
        norm_layer = get_norm(norm)
        self.nonlinear = FCNet([v_dim + q_dim, num_hid, num_hid], dropout= dropout, norm= norm, act= act)
        self.nonlinear_gate = FCNet([v_dim + q_dim, num_hid, num_hid], dropout= dropout, norm= norm, act= 'Sigmoid')
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        gate = self.nonlinear_gate(vq)
        logits = joint_repr*gate
        logits = self.linear(logits)
        return logits


# concat w/ 1 layer seperate, element-wise *, output layer
class Att_P(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, norm, act, dropout=0.0):
        super(Att_P, self).__init__()
        norm_layer = get_norm(norm)

        self.gated_tanh = GTH( in_dim= v_dim + q_dim, out_dim= num_hid, dropout= dropout, norm= norm, act= act)
        self.linear = norm_layer(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.gated_tanh(vq)
        logits = self.linear(joint_repr)
        return logits
