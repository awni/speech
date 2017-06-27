from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import random
import scipy.signal
import torch.utils.data as tud

from speech.utils import wave

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, batch_size):

        data = read_data_json(data_json)
        chars = set(t for d in data for t in d['text'])
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}

        for d in data:
            d['label'] = self.encode(d['text'])

        data.sort(key=lambda x : x['duration'])

        it_end = len(data) - batch_size + 1
        batch_idxs = [list(range(i, i+batch_size))
                for i in range(0, it_end, batch_size)]
        random.shuffle(batch_idxs)
        self.idxs = [i for b in batch_idxs for i in b]

        self.data = data
        self.output_dim = len(chars)
        self.batch_size = batch_size

    def encode(self, text):
        return [self.char_to_int[t] for t in text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[self.idxs[idx]]
        inputs = specgram_from_file(datum['audio'])
        print("IDX", idx, "SHAPE", inputs.shape)
        targets = self.encode(datum['text'])
        return inputs, targets

    def compute_mean_std(self, max_samples=5):
        idxs = range(len(self.data))
        random.shuffle(idxs)
        samples = [self[idx][0]
                   for idx in idxs[:max_samples]]
        samples = np.hstack(samples)
        mean = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        return mean, std

def zero_pad_concat(inputs):
    # Assumes last item in batch is the longest.
    shape = inputs[-1].shape
    shape = (len(inputs),shape[0], shape[1])
    cat_inputs = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        cat_inputs[e, :, :inp.shape[1]] = inp
    return cat_inputs

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = zero_pad_concat(inputs)
    # TODO convert inputs and labels to tensors.
    return inputs, labels

def make_loader(dataset, num_workers=0):
    sampler = tud.sampler.SequentialSampler(dataset)
    loader = tud.DataLoader(dataset,
                batch_size=dataset.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True)
    return loader

def specgram_from_file(audio_file):
    audio, sr = wave.array_from_wave(audio_file)
    return specgram(audio, sr)

def specgram(audio, sample_rate, window_size=20, step_size=10):
    ## TODO, compute LOG!! And viz some of these.
    nperseg = window_size * sample_rate / 1e3
    noverlap = step_size * sample_rate / 1e3
    _, _, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    return spec.astype(np.float32)

def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]

if __name__ == "__main__":
    dataset = AudioDataset("test.json")
