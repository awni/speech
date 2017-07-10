from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim

import speech.loader as loader
import speech.model

def run_epoch(model, optimizer, train_ldr, it, avg_loss):

    model_t = 0.0; data_t = 0.0
    end_t = time.time()
    for inputs, labels in train_ldr:
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        start_t = time.time()
        optimizer.zero_grad()
        out = model(inputs, labels)
        loss = model.loss(out, labels)

        loss.backward()

        g_norm = nn.utils.clip_grad_norm(model.opt_params(), 200)

        optimizer.step()
        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t

        exp_w = 0.95
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]
        msg = ("Iter: {}, Loss: {:.3f}, Loss Avg: {:.3f}, Grad Norm "
               "{:.3f}, Model Time {:.2f} (s), Data Time {:.2f} (s)")
        print(msg.format(it, loss.data[0], avg_loss,
                         g_norm, model_t, data_t))
        it += 1

    return it, avg_loss

def eval_dev(model, ldr):
    tot_loss = []
    for inputs, labels in ldr:
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        out = model(inputs, labels)
        loss = model.loss(out, labels)
        tot_loss.append(loss.data[0])
    avg_loss = sum(tot_loss) / len(tot_loss)
    print("Dev Loss: {:.2f}".format(avg_loss))

def run(config):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]

    # Datasets
    batch_size = opt_cfg["batch_size"]
    train_set = loader.AudioDataset(data_cfg["train_set"],
                    batch_size)
    char_map = train_set.int_to_char
    dev_set = loader.AudioDataset(data_cfg["dev_set"],
                batch_size, int_to_char=char_map)

    # Loader
    train_ldr = loader.make_loader(train_set)
    mean, std = train_set.compute_mean_std()
    dev_ldr = loader.make_loader(dev_set)

    # Model
    model = speech.model.Model(train_set.input_dim,
                train_set.output_dim, mean, std, char_map)

    if use_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model.opt_params(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"])

    run_state = (0, 0)
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        run_state = run_epoch(model, optimizer, train_ldr, *run_state)
        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))
        eval_dev(model, dev_ldr)
        torch.save(model, config["save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")

    parser.add_argument("config",
        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    args = parser.parse_args()

    with open(args.config, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False

    run(config)
