from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import glob
import os
import re
import subprocess
import tqdm

from speech.utils import wave

DATASETS = {
    "train_si284" : ["wsj1/doc/indices/si_tr_s.ndx",
                    "wsj0/doc/indices/train/tr_s_wv1.ndx"],
    "eval_92" : ["wsj0/doc/indices/test/nvp/si_et_20.ndx"],
    "dev_93" : ["wsj1/doc/indices/h1_p0.ndx"]
}
DOT_PATHS = ["wsj0/transcrp/dots/*/*/*.dot",
             "wsj1/trans/wsj1/*/*/*.dot",
             "wsj0/si_et_20/*/*.dot"]
ALLOWED = set("abcdefghijklmnopqrstuvwxyz.' -")
REPLACE = {
    ".point" : "point",
    ".period": "period",
    "'single-quote": "single-quote",
    "'single-close-quote": "single-close-quote",
    "`single-quote" : "single-quote",
    "-hyphen": "hyphen",
    ")close_paren" : "close-paren",
    "(left(-paren)-": "left-",
    "." : "",
    "--dash" : "dash",
    "-dash" : "dash",
}

def load_text(wsj_base):
    transcripts = []
    dots = []
    for d in DOT_PATHS:
        dots.extend(glob.glob(os.path.join(wsj_base, d)))
    for f in dots:
        with open(f, 'r') as fid:
            transcripts.extend(l.strip() for l in fid)
    transcripts = (t.split() for t in transcripts)
    # Key text by utterance id
    transcripts = {t[-1][1:-1] : clean(" ".join(t[:-1]))
                    for t in transcripts}
    return transcripts

def load_waves(wsj_base, files):
    waves = []
    for f in files:
        flist = os.path.join(wsj_base, f)
        with open(flist, 'r') as fid:
            lines = (l.split(":")[1].strip().strip("/")
                     for l in fid if l[0] != ';')
            lines = (os.path.join(wsj_base, l) for l in lines)
            # Replace wv1 with wav
            lines = (os.path.splitext(l)[0] + ".wav" for l in lines)
            waves.extend(sorted(lines))
    return waves

def clean(line):
    pl = line
    line = line.lower()
    line = re.sub("<|>|\\\\|\[\S+\]", "", line)
    toks = line.split()
    clean_toks = []
    for tok in toks:
        if re.match("\S+-dash", tok):
            clean_toks.extend(tok.split("-"))
        else:
            clean_toks.append(REPLACE.get(tok, tok))
    line = " ".join(t for t in clean_toks if t).strip()
    line = re.sub("\(\S*\)", "", line)
    line = re.sub("[()\*\":\?;!}{\~<>/&,\$\%\~]", "", line)
    line = re.sub("`", "'", line)
    line = " ".join(line.split())
    return line

def write_json(save_path, dataset, waves, transcripts):
    out_file = os.path.join(save_path, dataset + ".json")
    with open(out_file, 'w') as fid:
        for wave_file in tqdm.tqdm(waves):
            dur = wave.wav_duration(wave_file)
            key = os.path.basename(wave_file)
            key = os.path.splitext(key)[0]
            datum = {'text' : transcripts[key],
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

def convert_sph_to_wav(files):
    command = ["sph2pipe_v2.5/sph2pipe", "-p", "-f",
               "wav", "-c", "1"]
    for out_f in tqdm.tqdm(files):
        sph_f = os.path.splitext(out_f)[0] + ".wv1"
        subprocess.call(command + [sph_f, out_f])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess WSJ dataset.")
    parser.add_argument("wsj_base",
        help="Path where the dataset is stored.")
    parser.add_argument("save_path",
        help="Path to save dataset jsons.")
    parser.add_argument("--convert", action="store_true",
        help="Convert sphere to wav format.")
    args = parser.parse_args()

    transcripts = load_text(args.wsj_base)
    for d, v in DATASETS.items():
        waves = load_waves(args.wsj_base, v)
        if args.convert:
            print("Converting {}".format(d))
            convert_sph_to_wav(waves)
        if d == "train_si284":
            waves = filter(lambda x: "wsj0/si_tr_s/401" not in x, waves)
        print("Writing {}".format(d))
        write_json(args.save_path, d, waves, transcripts)

