
import numpy as np
import torch
import torch.autograd as autograd

import speech.models

import shared

def test_model():
    time_steps = 100
    freq_dim = 40
    batch_size = 4

    model = speech.models.Model(freq_dim, shared.model_config)

    x = torch.randn(batch_size, time_steps, freq_dim)
    x = autograd.Variable(x)

    x_enc = model.encode(x)
    t_dim = model.conv_out_size(time_steps, 0)
    expected_size = torch.Size((batch_size, t_dim, model.encoder_dim))

    # Check output size is correct.
    assert x_enc.size() == expected_size

    # Check cuda attribute works
    assert not model.is_cuda
    if torch.cuda.is_available():
        model.cuda()
        assert model.is_cuda
