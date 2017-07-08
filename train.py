from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import torch
import torch.autograd as autograd
import torch.optim

import speech.loader as loader
import speech.model


def run_epoch(model, optimizer, train_ldr, it):
    for inputs, labels in train_ldr:
        inputs = autograd.Variable(torch.FloatTensor(inputs))
        labels = autograd.Variable(torch.LongTensor(labels))
        optimizer.zero_grad()
        out = model(inputs, labels)
        loss = model.loss(out, labels)

        loss.backward()

        optimizer.step()

        # TODO, deal with the avg loss
        #exp_w = 0.9
        #avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]
        #if it % 100 == 0:
        print("Iter: {}, Loss: {:.3f}".format(it, loss.data[0]))
        it += 1
    return it

def run(config):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]

    # Datasets
    batch_size = opt_cfg["batch_size"]
    train_set = loader.AudioDataset(data_cfg["train_set"], batch_size)
    dev_set = loader.AudioDataset(data_cfg["dev_set"], batch_size)

    # Loader
    train_ldr = loader.make_loader(train_set)
    mean, std = train_set.compute_mean_std()

    # Model
    freq_dim = 161
    output_dim = train_set.output_dim  # TODO get these
    model = speech.model.Model(freq_dim, output_dim, mean, std)

    # Optimizer
    optimizer = torch.optim.SGD(model.opt_params(),
                    lr=opt_cfg["learning_rate"])

    it = 0
    for _ in range(opt_cfg["epochs"]):
        it = run_epoch(model, optimizer, train_ldr, it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")

    parser.add_argument("config",
        help="A json file with the training configuration.")
    args = parser.parse_args()
    with open(args.config, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    run(config)
