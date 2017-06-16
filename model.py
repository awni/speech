from __future__ import print_function
from __future__ import division

import math
import torch
import torch.nn as nn
import torch.autograd as autograd

class Model(nn.Module):

    def __init__(self, freq_dim, ouput_dim):
        super(Model, self).__init__()
        self.freq_dim = freq_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (32, 2), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, (32, 9), stride=(2, 2), padding=0),
            nn.ReLU()
        )

        self.rnn = nn.GRU(input_size=self.conv_out_dim,
                          hidden_size=128, num_layers=1,
                          batch_first=True, dropout=False,
                          bidirectional=False)

        self.fc = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim),
                nn.Softmax()
        )

    @property
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

    def forward(self, x):
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

        x = x.contiguous().view((b * t, -1))
        x = self.fc(x)
        x = x.view((b, t, -1))

        return x

if __name__ == "__main__":

    time_steps = 200
    freq_dim = 90
    output_dim = 10
    batch_size = 4

    x = autograd.Variable(torch.randn(batch_size, time_steps, freq_dim))

    model = Model(freq_dim, output_dim)
    output = model.forward(x)
    print(output.size())

