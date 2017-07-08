from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import time
import torch
import torch.autograd as autograd
import torch.optim

import speech.loader as loader
import speech.model


def run_epoch(model, optimizer, train_ldr, it, avg_loss):
    for inputs, labels in train_ldr:
        inputs = autograd.Variable(torch.from_numpy(inputs))
        labels = autograd.Variable(torch.from_numpy(labels))
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        out = model(inputs, labels)
        loss = model.loss(out, labels)

        loss.backward()

        optimizer.step()

        exp_w = 0.9
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]
        #if it % 100 == 0:
        msg = "Iter: {}, Loss: {:.3f}, Loss Avg: {:.3f}"
        print(msg.format(it, loss.data[0], avg_loss))
        it += 1
    return it, avg_loss

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
    model = speech.model.Model(train_set.input_dim,
                train_set.output_dim, mean, std)

    if use_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model.opt_params(),
                    lr=opt_cfg["learning_rate"])

    run_state = (0, 0)
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        run_state = run_epoch(model, optimizer, train_ldr, *run_state)
        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))
        torch.save(model.state_dict(), config["save_path"])


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

    use_cuda = torch.cuda.is_available()

    run(config)
