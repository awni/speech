from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn

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
        input_size = 32 * self.conv_out_size(freq_dim, 1)
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=rnn_dim,
                          num_layers=config["encoder_layers"],
                          batch_first=True, dropout=False,
                          bidirectional=False)

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution
                k = c.kernel_size[dim]
                s = c.stride[dim]
                n = (n - k + 1) / s
                n = int(math.ceil(n))
        return n

    def forward(self, batch):
        """
        Must be overriden by subclasses.
        """
        raise NotImplementedError

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

    def loss(self, x, y):
        """
        Must be overriden by subclasses.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return self.parameters().next().is_cuda

class LinearND(nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)

    def forward(self, x):
        size = x.size()
        n = np.prod(size[:-1])
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

