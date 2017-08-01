from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):

    def __init__(self, freq_dim, output_dim, config):
        super(Model, self).__init__()
        self.freq_dim = freq_dim
        self.output_dim = output_dim

        # For encoding
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (5, 32), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 32), stride=(2, 2), padding=0),
            nn.ReLU()
        )

        rnn_dim = config["rnn_dim"]
        self.rnn = nn.GRU(input_size=self.conv_out_dim(),
                          hidden_size=rnn_dim,
                          num_layers=config["encoder_layers"],
                          batch_first=True, dropout=False,
                          bidirectional=False)

        # For decoding
        self.embedding = nn.Embedding(self.output_dim, rnn_dim)
        self.dec_rnn = nn.GRUCell(rnn_dim, rnn_dim)
        self.h_init = nn.Parameter(data=torch.zeros(1, rnn_dim))
        self.attend = Attention()
        self.fc = nn.Linear(rnn_dim, self.output_dim)

    def cpu(self):
        super(Model, self).cpu()
        self.dec_rnn.bias_hh.data.squeeze_()
        self.dec_rnn.bias_ih.data.squeeze_()

    def conv_out_dim(self):
        dim = self.freq_dim
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution
                k = c.kernel_size[1]
                s = c.stride[1]
                dim = (dim - k + 1) / s
                dim = int(math.ceil(dim))
                channels = c.out_channels
        return dim * channels

    def forward(self, x, y):
        x = self.encode(x)
        out, _ = self.decode(x, y)
        return out

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)

        # At this point x should have shape
        # (batch, channels, time, freq)
        x = torch.transpose(x, 1, 2).contiguous()

        # Reshape x to be (batch, time, freq * channels)
        # for the RNN
        b, t, f, c = x.size()
        x = x.view((b, t, f * c))
        x, h = self.rnn(x)
        return x

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
            hx = self.dec_rnn(ix + sx, hx + sx)
            hx = hx + sx
            aligns.append(ax)
            out.append(hx)

        out = torch.stack(out, dim=1)
        b, t, d = out.size()
        out = out.view((b * t, d))
        out = self.fc(out)
        out = out.view((b, t, self.output_dim))

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
        hx = self.dec_rnn(ix + sx, hx + sx)

        out = hx + sx
        out = self.fc(out)

        return out, hx

    def loss(self, x, y):
        """
        x should be shape (batch, input sequence length, output_dim)
        y should be shape (batch, label sequence length)
        """
        batch_size, _, out_dim = x.size()
        x = x.view((-1, out_dim))
        y = y[:,1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(x, y, size_average=False)
        loss = loss / batch_size
        return loss

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

