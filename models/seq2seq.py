from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

from . import model

class Seq2Seq(model.Model):

    def __init__(self, freq_dim, vocab_size, config):
        super(Seq2Seq, self).__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRU(input_size=embed_dim,
                              hidden_size=rnn_dim,
                              num_layers=decoder_cfg["layers"],
                              batch_first=True, dropout=False)

        self.attend = Attention()

        # *NB* we predict vocab_size - 1 classes since we
        # never need to predict the start of sequence token.
        self.fc = model.LinearND(rnn_dim, vocab_size - 1)

    def loss(self, out, batch):
        _, y = self.collate(*batch)
        if self.is_cuda:
            y = y.cuda()

        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:,1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(out, y,
                size_average=False)
        loss = loss / batch_size
        return loss

    def forward(self, batch):
        x, y = self.collate(*batch)
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()

        x = self.encode(x)
        out, _ = self.decode(x, y)
        return out

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        inputs = self.embedding(y[:, :-1])

        out = []; aligns = []
        ax = None; hx = None
        for t in range(y.size()[1] - 1):
            ix = inputs[:, t:t+1, :]
            ox, hx = self.dec_rnn(ix, hx=hx)
            sx, ax = self.attend(x, ox, ax)
            aligns.append(ax)
            out.append(ox + sx)

        out = torch.cat(out, dim=1)
        out = self.fc(out)

        aligns = torch.cat(aligns, dim=0)
        return out, aligns

    def predict(self, probs):
        _, argmaxs = probs.max(dim=2)
        if argmaxs.is_cuda:
            argmaxs = argmaxs.cpu()
        argmaxs = argmaxs.data.numpy()
        return [seq.tolist() for seq in argmaxs]

    def decode_step(self, x, y, hx=None):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        ix = self.embedding(y)
        ox, hx = self.dec_rnn(ix, hx=hx)
        sx, _ = self.attend(x, ox) # TODO, have to save alis
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        return out, hx

    def collate(self, inputs, labels):
        inputs = model.zero_pad_concat(inputs)
        labels = end_pad_concat(labels)
        inputs = autograd.Variable(torch.from_numpy(inputs))
        labels = autograd.Variable(torch.from_numpy(labels))
        return inputs, labels

def end_pad_concat(labels):
    # Assumes last item in each example is the end token.
    batch_size = len(labels)
    end_tok = labels[0][-1]
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len),
                    fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    return cat_labels

class Attention(nn.Module):

    def __init__(self, kernel_size=11):
        """
        Module which Performs a single attention step along the
        second axis of a given encoded input. The module uses
        both 'content' and 'location' based attention.

        The 'content' based attention is an inner product of the
        decoder hidden state with each time-step of the encoder
        state.

        The 'location' based attention performs a 1D convollution
        on the previous attention vector and adds this into the
        next attention vector prior to normalization.

        *NB* Should compute attention differently if using cuda or cpu
        based on performance. See
        https://gist.github.com/awni/9989dd31642d42405903dec8ab91d1f0
        """
        super(Attention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)

    def forward(self, eh, dhx, ax=None):
        """
        Arguments:
            eh (FloatTensor): the encoder hidden state with
                shape (batch size, time, hidden dimension).
            dhx (FloatTensor): one time step of the decoder hidden
                state with shape (batch size, hidden dimension).
                The hidden dimension must match that of the
                encoder state.
            ax (FloatTensor): one time step of the attention
                vector.

        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        pax = torch.sum(eh * dhx, dim=2)
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).squeeze(dim=1)
            pax = pax + ax
        ax = nn.functional.softmax(pax)

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax
