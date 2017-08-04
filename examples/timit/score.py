from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import editdistance
import json

import preprocess

def remap(data):
    _, m48_39 = preprocess.load_phone_map()
    for d in data:
        d['prediction'] = [m48_39[p] for p in d['prediction']]
        d['label'] = [m48_39[p] for p in d['label']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CER on Timit with reduced phoneme set.")

    parser.add_argument("data_json",
        help="JSON with the transcripts.")
    args = parser.parse_args()

    with open(args.data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]

    remap(data)
    dist = sum(editdistance.eval(d['label'], d['prediction'])
                for d in data)
    total = sum(len(d['label']) for d in data)
    print("CER {:.3f}".format(dist / total))



