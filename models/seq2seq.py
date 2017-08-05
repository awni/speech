from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

from . import model

class Seq2Seq(model.Model):

    def __init__(self, freq_dim, output_dim, config):
        super(Seq2Seq, self).__init__(freq_dim, config)

        # For decoding
        rnn_dim = self.encoder_dim
        self.embedding = nn.Embedding(output_dim, rnn_dim)
        self.dec_rnn = nn.GRUCell(rnn_dim, rnn_dim)
        self.h_init = nn.Parameter(data=torch.zeros(1, rnn_dim))
        self.attend = Attention()
        self.fc = model.LinearND(rnn_dim, output_dim)

    def loss(self, out, batch):
        _, y = collate(*batch)
        if self.is_cuda:
            y = y.cuda()

        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:,1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(out, y, size_average=False)
        loss = loss / batch_size
        return loss

    def cpu(self):
        super(Model, self).cpu()
        self.dec_rnn.bias_hh.data.squeeze_()
        self.dec_rnn.bias_ih.data.squeeze_()

    def forward(self, batch):
        x, y = collate(*batch)
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()

        x = self.encode(x)
        out, _ = self.decode(x, y)
        return out

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        batch_size, seq_len = y.size()
        inputs = self.embedding(y[:, :-1])

        hx = self.h_init.expand(batch_size, self.h_init.size()[1])

        out = []; aligns = []
        for t in range(seq_len - 1):
            ix = inputs[:, t, :].squeeze(dim=1)
            sx, ax = self.attend(x, hx)
            hx = self.dec_rnn(ix, hx + sx)
            aligns.append(ax)
            out.append(hx + sx)

        out = torch.stack(out, dim=1)
        out = self.fc(out)

        aligns = torch.cat(aligns, dim=0)
        return out, aligns

    def decode_step(self, x, y, hx=None):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        batch_size = x.size()[0]

        if hx is None:
            hx = self.h_init.expand(batch_size, self.h_init.size()[1])

        ix = self.embedding(y)
        sx, _ = self.attend(x, hx)
        hx = self.dec_rnn(ix, hx + sx)

        out = hx + sx
        out = self.fc(out)

        return out, hx

def end_pad_concat(labels):
    # Assumes last item in each example is the end token.
    batch_size = len(labels)
    end_tok = labels[0][-1]
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len),
                    fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    return cat_labels

def collate(inputs, labels):
    inputs = model.zero_pad_concat(inputs)
    labels = end_pad_concat(labels)
    inputs = autograd.Variable(torch.from_numpy(inputs))
    labels = autograd.Variable(torch.from_numpy(labels))
    return inputs, labels

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, eh, dhx):
        """
        eh : the encoder hidden state with shape
        (batch size, time, hidden dimension).
        dhx : one time step of the decoder hidden state with shape
        (batch size, hidden dimension). The hidden dimension must
        match that of the encoder state.
        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        dhx = dhx.unsqueeze(1)
        ax = torch.bmm(eh, dhx.transpose(1,2))
        ax = nn.functional.softmax(ax.squeeze(dim=2))

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.bmm(eh.transpose(1,2), sx)
        return sx.squeeze(dim=2), ax
