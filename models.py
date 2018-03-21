import torch
import torch.nn as nn
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH


# Dropout p: probability of an element to be zeroed. Default: 0.5

"""
Name: Model

Pre written
"""
class Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]


        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class Model_2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier):
        super(Model_2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]

        att_1 = self.v_att_1(v, q_emb) # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


class Model_3(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier):
        super(Model_3, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.v_att_3 = v_att_3
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]

        att_1 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_3 = self.v_att_3(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2 + att_3
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

# Attn: 1 layer attention, output layer, softmax
def build_baseline(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_0(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 2 layer attention, output layer, softmax
def build_model_A1(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_1(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_2(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
def build_model_A3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_3(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, 1 layer, output layer, sigmoid
def build_model_A3S(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_3S(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
def build_model_A3x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_2(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_2(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier)


def build_model_A23P(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_P(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)

# 3*Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
def build_model_A3x3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)

# 3*Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2x3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)


# Attn: 1 layer seperate, element-wise *, output layer
def build_model_AP(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, output layer
def build_model_APx2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    v_att_2 = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_2(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier)


# Attn: 2 layer seperate, element-wise *, output layer
def build_model_APD(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_PD(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_AP_PC(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1=300, hid_dim_2=2048, out_dim=dataset.num_ans_candidates, dropout=dropC, norm=norm,
        act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_P_exact(dataset, num_hid, dropout, norm, activation):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=0.0)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=0, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = GTH(q_emb.num_hid, num_hid, dropout=0, norm=norm, act=activation)
    v_net = GTH(dataset.v_dim, num_hid, dropout=0, norm=norm, act=activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1= 300, hid_dim_2= 2048, out_dim=dataset.num_ans_candidates, dropout=0, norm=norm, act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_P_mod(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = GTH(q_emb.num_hid, num_hid, dropout=dropL, norm=norm, act=activation)
    v_net = GTH(dataset.v_dim, num_hid, dropout=dropL, norm=norm, act=activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1= 300, hid_dim_2= 2048, out_dim=dataset.num_ans_candidates, dropout=dropC, norm=norm, act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)
