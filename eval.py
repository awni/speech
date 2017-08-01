from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import tqdm

import speech.loader as loader
import speech.model

def eval_loop(model, ldr, use_cuda=False):
    losses = []
    corr = 0; tot = 0
    for inputs, labels in tqdm.tqdm(ldr):
        inputs.volatile = True
        labels.volatile = True
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        out = model(inputs, labels)
        loss = model.loss(out, labels)
        _, argmaxs = out.max(dim=2)
        corr += torch.sum((argmaxs == labels[:,1:]).data)
        tot += labels.numel()
        losses.append(loss.data[0])
    avg_loss = sum(losses) / len(losses)
    avg_acc = corr / tot
    return avg_loss, avg_acc

def run(model_path, dataset_json):

    use_cuda = torch.cuda.is_available()

    batch_size = 8

    model, preproc = speech.load(model_path)
    ldr = loader.make_loader(dataset_json,
            preproc, batch_size)

    model.cuda() if use_cuda else model.cpu()

    avg_loss, avg_acc = eval_loop(model, ldr, use_cuda=use_cuda)
    print("Avg Loss: {:.2f}, Avg Acc {:.2f}".format(avg_loss, avg_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    args = parser.parse_args()

    run(args.model, args.dataset)
