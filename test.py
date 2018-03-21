from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from base_model import Model
from loader import Data_loader

def test(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(args.bsize, args.emb, args.multilabel, train=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))

    model = Model(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)

    model = model.cuda()

    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise SystemExit('Need to provide model path.')

    result = []
    for step in xrange(loader.n_batches):
        # Batch preparation
        q_batch, a_batch, i_batch = loader.next_batch()
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        q_batch, i_batch = q_batch.cuda(), i_batch.cuda()

        # Do one model forward and optimize
        output = model(q_batch, i_batch)
        _, ix = output.data.max(1)
        for i, qid in enumerate(a_batch):
            result.append({
                'question_id': qid,
                'answer': loader.a_itow[ix[i]]
            })

    json.dump(result, open('result.json', 'w'))
    print ('Validation done')
