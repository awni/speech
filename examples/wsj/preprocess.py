from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import tqdm

from speech.utils import wave

datasets = ["test_dev93", "test_eval92", "train_si284"]

ALLOWED = set("abcdefghijklmnopqrstuvwxyz.' -")
REPLACE = {
    ".period": "period",
    "'single-quote": "single-quote",
    "-hyphen": "hyphen",
    ")paren" : "paren",
    ")end-the-paren" : "end-the-paren",
    ")end-of-paren" : "end-of-paren",
    "(parenthetically" : "parenthetically",
    "(parentheses" : "parentheses",
    "(left-paren": "left-paren",
    "(begin-parens" : "begin-parens",
    ")end-parens" : "end-parens",
    ")right-paren": "right-paren",
    "(in-parenthesis" : "in-parenthesis",
    "(paren" : "paren",
    ")close_paren" : "close-paren",
    ")close-paren" : "close-paren",
    ")un-parentheses" : "un-parentheses",
    "<noise>" : "",
    "-dash" : "dash",
}

def load_text(dataset):
    with open(dataset + ".txt", 'r') as fid:
        text = (l.strip().split() for l in fid)
        text = {l[0] : clean(" ".join(l[1:])) for l in text}
    return text

def load_waves(dataset):
    with open(dataset + ".flist", 'r') as fid:
        waves = [l.strip() for l in fid]
    return waves

def clean(line):

    line = line.lower()
    toks = [REPLACE.get(tok, tok) for tok in line.split()]
    line = " ".join(t for t in toks if t).strip()

    line = re.sub("[\*\":\?;!}{\~<>/&,]", "", line)
    line = re.sub("`", "'", line)
    line = re.sub("\(.*\)", "", line)
    line = " ".join(line.split())
    return line

def write_json(dataset, waves, text):
    with open(dataset + ".json", 'w') as fid:
        for wave_file in tqdm.tqdm(waves):
            dur = wave.wav_duration(wave_file)
            key = os.path.basename(wave_file)
            key = os.path.splitext(key)[0]
            datum = {'text' : text[key],
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

if __name__ == "__main__":
    for d in datasets:
        print("Writing {}".format(d))
        text = load_text(d)
        waves = load_waves(d)
        write_json(d, waves, text)

