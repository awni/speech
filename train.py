from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import torch

import speech.loader
import speech.model


def run_epoch(model, train_ldr):
    for example in train_ldr:
        print(len(example))

    print("RUNNING EPOCH")

def run(args):

    batch_size = 4
    train_set = speech.loader.AudioDataset(args.train, batch_size)

    epochs = 1
    freq_dim = 161
    output_dim = train_set.output_dim
    mean, std = train_set.compute_mean_std()
    train_ldr = speech.loader.make_loader(train_set)


    model = speech.model.Model(freq_dim, output_dim) # TODO get these
    run_epoch(model, train_ldr)

    mean, std = train_set.compute_mean_std()
    # TODO, set the model mean and std here.

    #dev_set = loader.AudioDataset(args.dev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")

    parser.add_argument("train",
        help="A json file with the training set.")
    #parser.add_argument("dev", default=None
    #    help="A json file with the dev set.")
    args = parser.parse_args()

    random.seed(2013)
    torch.manual_seed(2018)

    run(args)
