from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.autograd as autograd

from . import model

class Seq2Seq(model.Model):

    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRUCell(input_size=embed_dim,
                                  hidden_size=rnn_dim)

        self.attend = NNAttention(rnn_dim, log_t=decoder_cfg.get("log_t", False))

        self.sample_prob = decoder_cfg.get("sample_prob", 0)
        self.scheduled_sampling = (self.sample_prob != 0)

        # *NB* we predict vocab_size - 1 classes since we
        # never need to predict the start of sequence token.
        self.fc = model.LinearND(rnn_dim, vocab_size - 1)

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()
        self.volatile = True
        self.scheduled_sampling = False

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()
        self.volatile = False
        self.scheduled_sampling = (self.sample_prob != 0)

    def loss(self, batch):
        x, y = self.collate(*batch)
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        out, alis = self.forward_impl(x, y)
        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:,1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(out, y,
                size_average=False)
        loss = loss / batch_size
        return loss

    def forward_impl(self, x, y):
        x = self.encode(x)
        out, alis = self.decode(x, y)
        return out, alis

    def forward(self, batch):
        x, y = self.collate(*batch)
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        return self.forward_impl(x, y)[0]

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """

        inputs = self.embedding(y[:, :-1])

        out = []; aligns = []

        hx = torch.zeros((x.shape[0], x.shape[2]), requires_grad=False)
        if self.is_cuda:
            hx.cuda()
        ax = None; sx = None;
        for t in range(y.size()[1] - 1):
            sample = (out and self.scheduled_sampling)
            if sample and random.random() < self.sample_prob:
                ix = torch.max(out[-1], dim=2)[1]
                ix = self.embedding(ix)
            else:
                ix = inputs[:, t:t+1, :]

            if sx is not None:
                ix = ix + sx

            hx = self.dec_rnn(ix.squeeze(dim=1), hx)
            ox = hx.unsqueeze(dim=1)

            sx, ax = self.attend(x, ox, ax)
            aligns.append(ax)
            out.append(self.fc(ox + sx))

        out = torch.cat(out, dim=1)
        aligns = torch.stack(aligns, dim=1)
        return out, aligns

    def decode_step(self, x, y, state=None, softmax=False):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        if state is None:
            hx = torch.zeros((x.shape[0], x.shape[2]), requires_grad=False)
            if self.is_cuda:
                hx.cuda()
            ax = None; sx = None;
        else:
            hx, ax, sx = state

        ix = self.embedding(y)
        if sx is not None:
            ix = ix + sx
        hx = self.dec_rnn(ix.squeeze(dim=1), hx=hx)
        ox = hx.unsqueeze(dim=1)
        sx, ax = self.attend(x, ox, ax=ax)
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        if softmax:
            out = nn.functional.log_softmax(out, dim=1)
        return out, (hx, ax, sx)

    def predict(self, batch):
        probs = self(batch)
        argmaxs = torch.max(probs, dim=2)[1]
        argmaxs = argmaxs.cpu().data.numpy()
        return [seq.tolist() for seq in argmaxs]

    def infer_decode(self, x, y, end_tok, max_len):
        probs = []
        argmaxs = [y]
        state = None
        for e in range(max_len):
            out, state = self.decode_step(x, y, state=state)
            probs.append(out)
            y = torch.max(out, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            if torch.sum(y.data == end_tok) == y.numel():
                break

        probs = torch.cat(probs)
        argmaxs = torch.cat(argmaxs, dim=1)
        return probs, argmaxs

    def infer(self, batch, max_len=200):
        """
        Infer a likely output. No beam search yet.
        """
        x, y = self.collate(*batch)
        end_tok = y.data[0, -1] # TODO
        t = y
        if self.is_cuda:
            x = x.cuda()
            t = y.cuda()
        x = self.encode(x)

        # needs to be the start token, TODO
        y = t[:, 0:1]
        _, argmaxs = self.infer_decode(x, y, end_tok, max_len)
        argmaxs = argmaxs.cpu().data.numpy()
        return [seq.tolist() for seq in argmaxs]

    def beam_search(self, batch, beam_size=10, max_len=200):
        x, y = self.collate(*batch)
        start_tok = y.data[0, 0]
        end_tok = y.data[0, -1] # TODO
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        x = self.encode(x)

        y = y[:, 0:1].clone()

        beam = [((start_tok,), 0, None)];
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, state in beam:

                y[0] = hyp[-1]
                out, state = self.decode_step(x, y, state=state, softmax=True)
                out = out.cpu().data.numpy().squeeze(axis=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, state))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

            # Remove complete hypotheses
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)

            beam = filter(lambda x : x[0][-1] != end_tok, new_beam)
            beam = beam[:beam_size]

            if len(beam) == 0:
                break

            # Stopping criteria:
            # complete contains beam_size more probable
            # candidates than anything left in the beam
            if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
                break

        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        if len(complete) == 0:
            complete = beam
        hyp, score, _ = complete[0]
        return [hyp]

    def collate(self, inputs, labels):
        inputs = model.zero_pad_concat(inputs)
        labels = end_pad_concat(labels)
        inputs = autograd.Variable(torch.from_numpy(inputs))
        labels = autograd.Variable(torch.from_numpy(labels))
        if self.volatile:
            inputs.volatile = True
            labels.volatile = True
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

    def __init__(self, kernel_size=11, log_t=False):
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
        self.log_t = log_t

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
        # location attention
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)


        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).squeeze(dim=1)
            pax = pax + ax

        if self.log_t:
            log_t = math.log(pax.size()[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax,  dim=1)

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax

class ProdAttention(nn.Module):

    def __init__(self):
        super(ProdAttention, self).__init__()

    def forward(self, eh, dhx, ax=None):
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)

        ax = nn.functional.softmax(pax, dim=1)

        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax

class NNAttention(nn.Module):

    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(NNAttention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(
                     nn.ReLU(),
                     model.LinearND(n_channels, 1))
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        pax = eh + dhx
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).transpose(1, 2)
            pax = pax + ax

        pax = self.nn(pax)
        pax = pax.squeeze(dim=2)
        if self.log_t:
            log_t = math.log(pax.size()[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)

        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax
