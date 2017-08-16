
import numpy as np
import torch
import torch.autograd as autograd

from speech.models import Seq2Seq

import shared

def test_model():
    freq_dim = 120
    vocab_size = 10

    np.random.seed(1337)
    torch.manual_seed(1337)

    conf = shared.model_config
    conf["decoder"] = {"embedding_dim" : 8,
                       "layers" : 2}
    model = Seq2Seq(freq_dim, vocab_size + 1, conf)
    batch = shared.gen_fake_data(freq_dim, vocab_size)
    batch_size = len(batch[0])

    out = model(batch)
    loss = model.loss(out, batch)

    assert out.size()[0] == batch_size
    assert out.size()[2] == vocab_size
    assert len(out.size()) == 3

    x, y = model.collate(*batch)
    x_enc = model.encode(x)

    state = None
    out_s = []
    for t in range(y.size()[1] - 1):
        ox, state = model.decode_step(x_enc, y[:,t:t+1], state=state)
        out_s.append(ox)
    out_s = torch.stack(out_s, dim=1)
    assert out.size() == out_s.size()
    assert np.allclose(out_s.data.numpy(),
                       out.data.numpy(),
                       rtol=1e-5,
                       atol=1e-7)

