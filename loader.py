from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import random
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud

from speech.utils import wave

class Preprocessor():

    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, max_samples=100):
        """
        Builds a preprocessor from a dataset.
        data_json is a file containing a json representation of each example
        per line.
        max_samples is the maximum number of examples to be used
        in computing summary statistics.
        """
        data = read_data_json(data_json)

        # Compute data mean, std from sample
        audio_files = [d['audio'] for d in data]
        random.shuffle(audio_files)
        self.mean, self.std = compute_mean_std(audio_files[:max_samples])

        # Make char map
        chars = set(t for d in data for t in d['text'])
        chars.update([self.END, self.START])
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}

    def encode(self, text):
        text = [self.START] + list(text) + [self.END]
        return [self.char_to_int[t] for t in text]

    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        idx = 0
        while idx < len(text) and text[idx] != self.END:
            idx += 1
        return "".join(text[:idx])

    def preprocess(self, wave_file, text):
        inputs = log_specgram_from_file(wave_file)
        inputs = (inputs - self.mean) / self.std
        targets = self.encode(text)
        return inputs, targets

    @property
    def input_dim(self):
        return 161 # TODO, awni, set this automatically

    @property
    def output_dim(self):
        return len(self.int_to_char)

def compute_mean_std(audio_files):
    samples = [log_specgram_from_file(af)
               for af in audio_files]
    samples = np.vstack(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):

        data = read_data_json(data_json)
        self.preproc = preproc

        # Sort by input length and make minibatches
        data.sort(key=lambda x : x['duration'])
        it_end = len(data) - batch_size + 1
        batch_idxs = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        random.shuffle(batch_idxs)
        self.idxs = [i for b in batch_idxs for i in b]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[self.idxs[idx]]
        datum = self.preproc.preprocess(datum["audio"],
                                        datum["text"])
        return datum

def zero_pad_concat(inputs):
    # Assumes last item in batch is the longest.
    shape = inputs[-1].shape
    shape = (len(inputs), shape[0], shape[1])
    cat_inputs = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        cat_inputs[e, :inp.shape[0], :] = inp
    return cat_inputs

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

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = zero_pad_concat(inputs)
    labels = end_pad_concat(labels)
    inputs = autograd.Variable(torch.from_numpy(inputs))
    labels = autograd.Variable(torch.from_numpy(labels))
    return inputs, labels

def make_loader(dataset_json, preproc,
                batch_size, num_workers=4):
    dataset = AudioDataset(dataset_json, preproc,
                           batch_size)
    sampler = tud.sampler.SequentialSampler(dataset)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True)
    return loader

def log_specgram_from_file(audio_file):
    audio, sr = wave.array_from_wave(audio_file)
    return log_specgram(audio, sr)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = window_size * sample_rate / 1e3
    noverlap = step_size * sample_rate / 1e3
    _, _, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]
