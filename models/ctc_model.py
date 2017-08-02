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
        super(CTC, self).__init__(freq_dim, output_dim, config)

        # For decoding
        rnn_dim = config["rnn_dim"]
        self.fc = model.LinearND(rnn_dim, output_dim)

    def forward(self, x):
        x = self.encode(x)
        return self.fc(x)

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(batch)
        if self.is_cuda:
            x = x.cuda()

        out = self(x)

        batch_size, _, out_dim = out.size()
        loss_fn = ctc.CTCLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def collate(self, batch):
        inputs, labels = zip(*batch)
        x_lens = (i.shape[0] for i in inputs)
        x_lens = [self.conv_out_size(i, 0) for i in x_lens]
        x_lens = torch.IntTensor(x_lens)
        x = torch.FloatTensor(zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        return [autograd.Variable(v) for v in batch]

def zero_pad_concat(inputs):
    # Assumes last item in batch is the longest.
    shape = inputs[-1].shape
    shape = (len(inputs), shape[0], shape[1])
    cat_inputs = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        cat_inputs[e, :inp.shape[0], :] = inp
    return cat_inputs


