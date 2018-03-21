import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, opt, wd):
    utils.create_dir(output)
    # Paper uses AdaDelta
    if opt == 'Adadelta':
        optim = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=wd)
    elif opt == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=wd, momentum=0, centered=False)
    else:
        optim = torch.optim.Adamax(model.parameters(), weight_decay=wd)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(1):

        num_runs=0

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda() # boxes not used
            q = Variable(q).cuda()
            a = Variable(a).cuda() # true labels

            pred = model(v, b, q, a)

            if num_runs ==1:
                return
            num_runs +=1
