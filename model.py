from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):

    def __init__(self, freq_dim, output_dim):
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

        rnn_dim = 128
        self.rnn = nn.GRU(input_size=self.conv_out_dim(),
                          hidden_size=rnn_dim, num_layers=1,
                          batch_first=True, dropout=False,
                          bidirectional=False)

        # For decoding
        embed_dim = 32
        self.embedding = nn.Embedding(self.output_dim, embed_dim)
        self.dec_rnn = nn.GRUCell(embed_dim, rnn_dim)
        self.h_init = nn.Parameter(data=torch.zeros(1, rnn_dim))
        self.attend = Attention()
        self.fc = nn.Linear(rnn_dim, self.output_dim)

    def cpu_patch(self):
        self.cpu()
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

    def forward(self, x, y=None):
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

        if y is not None:
            return self.decoder_train(x, y)

        return self.decoder_test(x)

    def decoder_train(self, x, y):
        """
        x should be shape (batch, time, dimension)
        y should be shape (batch, label sequence length)
        """
        batch_size, seq_len = y.size()
        inputs = self.embedding(y[:, :-1])

        hx = self.h_init.expand(batch_size, self.h_init.size()[1])

        sx = self.attend(x, hx)
        hx = hx + sx
        out = [hx]
        for t in range(seq_len - 1):
            ix = inputs[:, t, :].squeeze(dim=1)
            hx = self.dec_rnn(ix, hx)
            sx = self.attend(x, hx)
            hx = hx + sx
            out.append(hx)
        out = torch.stack(out, dim=1)
        b, t, d = out.size()
        out = out.view((b * t, d))
        out = self.fc(out)
        out = out.view((b, t, self.output_dim))
        return out

    def decoder_test(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        """
        x should be shape (batch, input sequence length, output_dim)
        y should be shape (batch, label sequence length)
        """
        batch_size, _, out_dim = x.size()
        x = x.view((-1, out_dim))
        y = y.contiguous().view(-1)
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
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        dhx = dhx.unsqueeze(1)
        sx = torch.bmm(eh, dhx.transpose(1,2))
        sx = nn.functional.softmax(sx.squeeze(dim=2))

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = sx.unsqueeze(2)
        sx = torch.bmm(eh.transpose(1,2), sx)
        return sx.squeeze(dim=2)
