from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):

    def __init__(self, freq_dim, output_dim, mean, std):
        super(Model, self).__init__()
        self.freq_dim = freq_dim
        self.output_dim = output_dim

        ## For encoding
        self.mean = nn.Parameter(data=torch.Tensor(mean),
                                 requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(std),
                                requires_grad=False)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (32, 2), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, (32, 9), stride=(2, 2), padding=0),
            nn.ReLU()
        )

        self.rnn = nn.GRU(input_size=self.conv_out_dim(),
                          hidden_size=128, num_layers=1,
                          batch_first=True, dropout=False,
                          bidirectional=False)

        # For decoding
        self.embedding = nn.Embedding(self.output_dim, 32)
        self.dec_rnn = nn.GRUCell(32, 128)
        self.ho = nn.Parameter(data=torch.zeros(1, 128),
                               requires_grad=False)
        self.attend = Attention()
        self.fc = nn.Linear(128, self.output_dim)


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

    def opt_params(self):
        return filter(lambda p : p.requires_grad, self.parameters())

    def forward(self, x, y=None):
        # TODO, awni, remove this when broadcasting is in stable pytorch
        x = broadcast_sub(x, self.mean)
        x = broadcast_div(x, self.std)

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

        x = x.contiguous().view((b, t, -1))

        if y is not None:
            return self.decode_train(x, y)

        return self.decode_test(x)

    def decode_train(self, x, y):
        # x should be shape (batch, time, dimension)
        # y should be shape (batch, label sequence length)
        batch_size, seq_len = y.size()
        inputs = self.embedding(y[:, :-1])
        hx = self.ho.expand(batch_size, self.ho.size()[1])
        out = []
        for t in range(seq_len - 1):
            ix = inputs[:, t, :].squeeze(dim=1)
            hx = self.dec_rnn(ix, hx)
            hx = self.attend(x, hx)
            out.append(hx)
        out = torch.stack(out, dim=1)
        b, t, d = out.size()
        out = out.view((b * t, d))
        out = self.fc(out)
        out = out.view((b, t, self.output_dim))
        return out

    def decode_test(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        # x should be shape (batch, sequence length, output_dim)
        # y should be shape (batch, sequence length)
        batch_size = x.size()[0]
        y = y[:, 1:]
        x = x.view((-1, self.output_dim))
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

def broadcast_sub(a, b):
    b = b.unsqueeze(0).unsqueeze(0)
    b = b.expand_as(a)
    return a.sub(b)

def broadcast_div(a, b):
    b = b.unsqueeze(0).unsqueeze(0)
    b = b.expand_as(a)
    return a.div(b)


