
import numpy as np
import torch
import torch.autograd as autograd

import speech.model

def test_model():
    time_steps = 200
    freq_dim = 90
    output_dim = 10
    batch_size = 4
    seq_len = 20


    mean = torch.randn(freq_dim)
    std = torch.randn(freq_dim)
    model = speech.model.Model(freq_dim, output_dim, mean, std)

    x = autograd.Variable(torch.randn(batch_size, time_steps, freq_dim))
    labels = np.random.randint(0, output_dim, (batch_size, seq_len))
    y = autograd.Variable(torch.LongTensor(labels))
    output = model.forward(x, y)

