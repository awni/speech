from __future__ import print_function
from __future__ import division

import argparse
import glob
import os

SETS = {
    "train" : ["train-clean-100"],
    "dev" : ["dev-clean"],
    "test" : []
    }

def load_transcripts(path, dataset):
    pattern = os.path.join(path, dataset, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip().split() for l in fid)
            lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("output_directory",
        help="The dataset is saved in <output_directory>/LibriSpeech.")
    args = parser.parse_args()

    path = os.path.join(args.output_directory, "LibriSpeech")

    for dataset, dirs in SETS.items():
        for d in dirs:
            print("Preprocessing {}".format(d))
            load_transcripts(path, d)
