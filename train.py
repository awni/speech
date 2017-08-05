from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import tensorboard_logger as tb
import time
import torch
import torch.nn as nn
import torch.optim
import tqdm

import speech.loader as loader
import speech.models as models
import speech.eval

def run_epoch(model, optimizer, train_ldr, it, avg_loss):

    model_t = 0.0; data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    for batch in tq:
        start_t = time.time()
        optimizer.zero_grad()
        out = model(batch)
        loss = model.loss(out, batch)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)

        optimizer.step()
        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t

        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss.data[0]
        tb.log_value('train_loss', loss.data[0], it)
        tq.set_postfix(iter=it, loss=loss.data[0],
                avg_loss=avg_loss, grad_norm=grad_norm,
                model_time=model_t, data_time=data_t)
        it += 1

    return it, avg_loss

def eval_dev(model, ldr):
    avg_loss, _, _ = speech.eval.eval_loop(model, ldr)
    print("Dev Loss: {:.2f}".format(avg_loss))
    return avg_loss

def run(config):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"],
                                  start_and_end=False)
    train_ldr = loader.make_loader(data_cfg["train_set"],
                        preproc, batch_size)
    dev_ldr = loader.make_loader(data_cfg["dev_set"],
                        preproc, batch_size)

    # Model
    model_class = eval("models." + model_cfg["class"])
    model = model_class(preproc.input_dim,
                        preproc.output_dim,
                        model_cfg)
    model.cuda() if use_cuda else model.cpu()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"])

    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()

        run_state = run_epoch(model, optimizer, train_ldr, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss = eval_dev(model, dev_ldr)

        # Log for tensorboard
        tb.log_value("dev_loss", dev_loss, e)

        speech.save(model, preproc, config["save_path"])

        # Save the best model on the dev set
        if dev_loss < best_so_far:
            best_so_far = dev_loss
            speech.save(model, preproc,
                    config["save_path"], tag="best")

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

    tb.configure(config["save_path"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config)
