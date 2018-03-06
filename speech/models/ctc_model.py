from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.autograd as autograd

import functions.ctc as ctc
from . import model
from .ctc_decoder import decode

class CTC(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super(CTC, self).__init__(freq_dim, config)

        # include the blank token
        self.blank = output_dim
        self.fc = model.LinearND(self.encoder_dim, output_dim + 1)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x)

    def forward_impl(self, x, softmax=False):
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x = self.fc(x)
        if softmax:
            return torch.nn.functional.softmax(x, dim=2)
        return x

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        out = self.forward_impl(x)

        loss_fn = ctc.CTCLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
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
        probs = self.forward_impl(x, softmax=True)
        probs = probs.data.cpu().numpy()
        return [decode(p, beam_size=1, blank=self.blank)[0]
                    for p in probs]

    @staticmethod
    def max_decode(pred, blank):
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq
