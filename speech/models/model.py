from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim

        encoder_cfg = config["encoder"]
        conv_cfg = encoder_cfg["conv"]

        convs = []
        in_c = 1
        for out_c, h, w, s in conv_cfg:
            conv = nn.Conv2d(in_c, out_c, (h, w),
                             stride=(s, s), padding=0)
            convs.extend([conv, nn.ReLU()])
            if config["dropout"] != 0:
                convs.append(nn.Dropout(p=config["dropout"]))
            in_c = out_c

        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(input_dim, 1)
        assert conv_out > 0, \
          "Convolutional ouptut frequency dimension is negative."

        rnn_cfg = encoder_cfg["rnn"]
        self.rnn = nn.GRU(input_size=conv_out,
                          hidden_size=rnn_cfg["dim"],
                          num_layers=rnn_cfg["layers"],
                          batch_first=True, dropout=config["dropout"],
                          bidirectional=rnn_cfg["bidirectional"])
        self._encoder_dim = rnn_cfg["dim"]

        self.volatile = False

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
        Must be overridden by subclasses.
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

        if self.rnn.bidirectional:
            half = x.size()[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]

        return x

    def loss(self, x, y):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()
        self.volatile = True

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()
        self.volatile = False

    def infer(self, x):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        return self._encoder_dim

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
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat

