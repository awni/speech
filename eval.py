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
    for inputs, labels in tqdm.tqdm(ldr):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        out = model(inputs, labels)
        loss = model.loss(out, labels)
        #seq = np.argmax(out.cpu().data.numpy(), axis=2).squeeze()
        #print(dataset.decode(seq))
        losses.append(loss.data[0])
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def run(model_path, dataset_json):

    use_cuda = torch.cuda.is_available()

    batch_size = 8

    model, preproc = speech.load(model_path)
    ldr = loader.make_loader(dataset_json,
            preproc, batch_size)

    if use_cuda:
        model.cuda()

    avg_loss = eval_loop(model, ldr, use_cuda=use_cuda)
    print("Average Loss: {:.2f}".format(avg_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    args = parser.parse_args()

    run(args.model, args.dataset)
