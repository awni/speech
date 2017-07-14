from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import tqdm

import speech.loader as loader
import speech.model

def run(model_path, dataset_json):

    batch_size = 8

    model = torch.load(model_path)
    dataset = loader.AudioDataset(dataset_json, batch_size,
                          int_to_char=model.char_map)

    ldr = loader.make_loader(dataset)

    if use_cuda:
        model.cuda()

    losses = []
    for inputs, labels in ldr:#tqdm.tqdm(ldr):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        out = model(inputs, labels)
        #seq = np.argmax(out.cpu().data.numpy(), axis=2).squeeze()
        #print(dataset.decode(seq))
        loss = model.loss(out, labels)
        print(loss.data[0])
        losses.append(loss.data[0])
    avg_loss = sum(losses) / len(losses)
    print("Dev Loss: {:.2f}".format(avg_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    import random
    random.seed(2017)
    run(args.model, args.dataset)
