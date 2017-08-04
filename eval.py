from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import editdistance
import json
import torch
import tqdm
import speech
import speech.loader as loader

def compute_cer(labels, preds):
    """
    Arguments:
        labels (list): list of grount truth sequences.
        preds (list): list of predicted sequences.
    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred)
                for label, pred in zip(labels, preds))
    total = sum(len(label) for label in labels)
    return dist / total

def eval_loop(model, ldr):
    losses = []
    all_preds = []; all_labels = []
    for batch in tqdm.tqdm(ldr):
        # TODO, handle volatile
        probs = model(batch)
        loss = model.loss(probs, batch)
        preds = model.predict(probs)
        losses.append(loss.data[0])
        all_preds.extend(preds)
        all_labels.extend(batch[1])
    cer = compute_cer(all_labels, all_preds)
    avg_loss = sum(losses) / len(losses)
    return avg_loss, cer, list(zip(all_labels, all_preds))

def run(model_path, dataset_json,
        batch_size=8, tag="best",
        out_file=None):

    use_cuda = torch.cuda.is_available()

    model, preproc = speech.load(model_path, tag=tag)
    ldr = loader.make_loader(dataset_json,
            preproc, batch_size)

    model.cuda() if use_cuda else model.cpu()

    avg_loss, cer, results = eval_loop(model, ldr)
    msg = "Avg Loss: {:.2f}, CER {:.3f}"
    print(msg.format(avg_loss, cer))

    if out_file is not None:
        with open(out_file, 'w') as fid:
            for label, pred in results:
                res = {'prediction' : preproc.decode(pred),
                       'label' : preproc.decode(label)}
                json.dump(res, fid)
                fid.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    parser.add_argument("--last", action="store_true",
        help="Last saved model instead of best on dev set.")
    parser.add_argument("--save",
        help="Optional file to save predicted results.")
    args = parser.parse_args()

    run(args.model, args.dataset,
        tag=None if args.last else "best",
        out_file=args.save)
