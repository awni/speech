from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

import transducer.decoders as td
import transducer.functions.transducer as transducer
from . import model

class Transducer(model.Model):
    def __init__(self, freq_dim, vocab_size, config):
        super(Transducer, self).__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRU(input_size=embed_dim,
                              hidden_size=rnn_dim,
                              num_layers=decoder_cfg["layers"],
                              batch_first=True, dropout=config["dropout"])

        # include the blank token
        self.blank = vocab_size
        self.fc1 = model.LinearND(rnn_dim, rnn_dim)
        self.fc2 = model.LinearND(rnn_dim, vocab_size + 1)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        return self.forward_impl(x, y_mat)

    def forward_impl(self, x, y):
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        x = self.encode(x)
        out = self.decode(x, y)
        return out

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        out = self.forward_impl(x, y_mat)
        loss_fn = transducer.TransducerLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        y = self.embedding(y)

        # preprend zeros
        b, t, h = y.shape
        start = autograd.Variable(torch.zeros((b, 1, h)))
        if self.is_cuda:
            start = start.cuda()
        y = torch.cat([start, y], dim=1)

        y, _ = self.dec_rnn(y)

        # Combine the input states and the output states
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=1)
        out = self.fc1(x) + self.fc1(y)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=3)
        return out

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        batch = [autograd.Variable(v) for v in batch]
        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch

    def infer(self, batch, beam_size=4):
        out = self(batch)
        out = out.cpu().data.numpy()
        preds = []
        for e, (i, l) in enumerate(zip(*batch)):
            T = i.shape[0]
            U = len(l) + 1
            lp = out[e, :T, :U, :]
            preds.append(td.decode_static(lp, beam_size, blank=self.blank)[0])
        return preds

    def label_collate(self, labels):
        # Doesn't matter what we pad the end with
        # since it will be ignored.
        batch_size = len(labels)
        end_tok = labels[0][-1]
        max_len = max(len(l) for l in labels)
        cat_labels = np.full((batch_size, max_len),
                        fill_value=end_tok, dtype=np.int64)
        for e, l in enumerate(labels):
            cat_labels[e, :len(l)] = l
        labels = autograd.Variable(torch.LongTensor(cat_labels))
        if self.volatile:
            labels.volatile = True
        return labels
