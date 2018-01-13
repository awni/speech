from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle
import json
import numpy as np
import torch
import tqdm

import speech
import speech.models as models
import speech.loader as loader
import speech.utils.ctc_align as ctc_align
import transducer.ref_transduce as rt

def to_sample_n(index):
    index += 4 # account for 2 layers of valid conv window size 5
    index *= 2 # account for stride
    index *= 10 # 10ms step size in specgram
    index *= 16 # 16 samples per millisecond
    return index

def onset_offset(alignment, stride=2):
    # (phone, onset (idx), offset(idx))
    timings = []
    start_idx = None
    for e, p in enumerate(alignment):
        if start_idx == None and p < 48:
            start_idx = e
            continue
        if e == 0:
            continue
        if p != alignment[e-1]:
            timings.append((alignment[e-1], start_idx, e-1))
            start_idx = e if p < 48 else None
    if p < 48:
        timings.append((p, start_idx, e))
    # Convert index -> samples
    timings = [(p, to_sample_n(i), to_sample_n(j))
                for p, i, j in timings]
    return timings

def eval_loop(model, dataset):
    alignments = []
    for datum in tqdm.tqdm(dataset.data):
        x, labels = dataset.preproc.preprocess(datum["audio"],
                                               datum["text"])
        res = model([[x],[labels]])
        if type(model) == models.Seq2Seq:
            attention = res[1].data.cpu().numpy().squeeze(axis=0)
            alignment = np.argmax(attention, axis=1).tolist()
            alignment = [(p, to_sample_n(i), to_sample_n(i))
                          for p, i in zip(labels[1:-1], alignment[:-1])]
        elif type(model) == models.CTC:
            probs = res.data.cpu().numpy().squeeze(axis=0)
            alignment = ctc_align.align(probs, labels)
            alignment = onset_offset(alignment)
        elif type(model) == models.Transducer:
            probs = res.data.cpu().numpy().squeeze(axis=0)
            alignment = rt.align(probs, labels, probs.shape[-1] - 1)
            alignment = [(p, to_sample_n(i), to_sample_n(i))
                          for p, i in zip(labels, alignment)]
        alignments.append((datum["audio"], alignment))
    return alignments

def run(model_path, dataset_json, out_file=None):
    batch_size = 1
    use_cuda = torch.cuda.is_available()

    model, preproc = speech.load(model_path, tag="best")
    dataset = loader.AudioDataset(dataset_json, preproc, batch_size)

    model.cuda() if use_cuda else model.cpu()
    model.set_eval()

    alignments = eval_loop(model, dataset)
    n_alignments = []
    for key, timings in alignments:
        timings = [(preproc.int_to_char[a], s, e)
                    for a, s, e in timings]
        n_alignments.append((key, timings))
    alignments = n_alignments
    if out_file is not None:
        with open(out_file, 'w') as fid:
            pickle.dump(alignments, fid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    parser.add_argument("--save",
        help="Optional file to save predicted results.")
    args = parser.parse_args()

    run(args.model, args.dataset, out_file=args.save)
