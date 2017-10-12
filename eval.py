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

def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
            predicted sequence pairs.

    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total

def eval_loop(model, ldr):
    all_preds = []; all_labels = []
    for batch in tqdm.tqdm(ldr):
        preds = model.infer(batch)
        all_preds.extend(preds)
        all_labels.extend(batch[1])
    return list(zip(all_labels, all_preds))

def run(model_path, dataset_json,
        batch_size=8, tag="best",
        out_file=None):

    use_cuda = torch.cuda.is_available()

    model, preproc = speech.load(model_path, tag=tag)
    ldr = loader.make_loader(dataset_json,
            preproc, batch_size)

    model.cuda() if use_cuda else model.cpu()
    model.set_volatile()
    model.eval()

    results = eval_loop(model, ldr)
    results = [(preproc.decode(label), preproc.decode(pred))
               for label, pred in results]
    cer = compute_cer(results)
    print("CER {:.3f}".format(cer))

    if out_file is not None:
        with open(out_file, 'w') as fid:
            for label, pred in results:
                res = {'prediction' : pred,
                       'label' : label}
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
