from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.autograd as autograd

import functions.ctc as ctc
from . import model

class CTC(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super(CTC, self).__init__(freq_dim, config)
        self.fc = model.LinearND(self.encoder_dim, output_dim)
        self.blank = output_dim - 1

    def training_loss(self, batch):
        out = self.forward(batch)
        return self.loss(batch)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x)

    def forward_impl(self, x):
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        return self.fc(x)

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        out = self.forward_impl(x)

        batch_size, _, out_dim = out.size()
        loss_fn = ctc.CTCLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def collate(self, inputs, labels):
        x_lens = (i.shape[0] for i in inputs)
        x_lens = [self.conv_out_size(i, 0) for i in x_lens]
        x_lens = torch.IntTensor(x_lens)
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        batch = [autograd.Variable(v) for v in batch]
        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch

    def infer(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs = self.forward_impl(x)

        _, argmaxs = probs.max(dim=2)
        if argmaxs.is_cuda:
            argmaxs = argmaxs.cpu()
        argmaxs = argmaxs.data.numpy()
        return [self.max_decode(seq, blank=self.blank)
                for seq in argmaxs]

    @staticmethod
    def max_decode(pred, blank=0):
        prev = pred[0]
        seq = [prev] if prev is not blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq
