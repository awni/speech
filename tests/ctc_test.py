
import torch
import torch.autograd as autograd

from speech.models import CTC

import shared

def test_ctc_model():
    freq_dim = 40
    output_dim = 10

    batch = shared.gen_fake_data(freq_dim, output_dim)
    batch_size = len(batch[0])

    model = CTC(freq_dim, output_dim, shared.model_config)
    out = model(batch)

    assert out.size()[0] == batch_size
    assert out.size()[2] == output_dim
    assert len(out.size()) == 3

    loss = model.loss(batch)
    preds = model.infer(batch)
    assert len(preds) == batch_size


def test_argmax_decode():
    pre = [1, 2, 2, 0, 0, 0, 2, 1]
    post = [1, 2, 2, 1]
    assert CTC.max_decode(pre) == post

    pre = [2, 2, 2]
    post = [2]
    assert CTC.max_decode(pre) == post

    pre = [0, 0, 0]
    post = []
    assert CTC.max_decode(pre) == post
