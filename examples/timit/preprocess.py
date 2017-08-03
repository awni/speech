from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import random
import tqdm

from speech.utils import data_helpers
from speech.utils import wave

WAV_EXT = "wv" # using wv since NIST took wav

def load_phone_map():
    with open("phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0] : l[1] for l in lines}
    m48_39 = {l[1] : l[2] for l in lines}
    return m60_48, m48_39

def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.phn")
    m60_48, _ = load_phone_map()
    files = glob.glob(pattern)
    # Standard practic is to remove all "sa" sentences
    # for each speaker since they are the same for all.
    filt_sa = lambda x : os.path.basename(x)[:2] != "sa"
    files = filter(filt_sa, files)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = (l.split()[-1] for l in lines)
            phonemes = [m60_48[p] for p in phonemes if p in m60_48]
            data[f] = phonemes
    return data

def split_by_speaker(data, dev_frac):

    def speaker_id(f):
        return os.path.basename(os.path.dirname(f))

    speaker_dict = collections.defaultdict(list)
    for k, v in data.items():
        speaker_dict[speaker_id(k)].append((k, v))
    speakers = speaker_dict.keys()
    random.shuffle(speakers)
    cut = int(len(speakers) * dev_frac)
    train, dev = speakers[cut:], speakers[:cut]
    train = dict(v for s in train for v in speaker_dict[s])
    dev = dict(v for s in dev for v in speaker_dict[s])
    return train, dev

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.wav",
            new_ext=WAV_EXT,
            use_avconv=False)

def build_json(data, path, set_name):
    basename = set_name + os.path.extsep + "json"
    with open(os.path.join(path, basename), 'w') as fid:
        for k, t in tqdm.tqdm(data.items()):
            wave_file = os.path.splitext(k)[0] + os.path.extsep + WAV_EXT
            dur = wave.wav_duration(wave_file)
            datum = {'text' : t,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess Timit dataset.")

    parser.add_argument("output_directory",
        help="Path where the dataset is saved.")
    parser.add_argument("--dev_frac", default=0.12, type=float,
        help="Fraction to hold out (by speaker) for the dev set.")
    args = parser.parse_args()

    path = os.path.join(args.output_directory, "timit")

    print("Converting files from NIST to standard wave format...")
    convert_to_wav(path)

    print("Preprocessing train")
    transcripts = load_transcripts(os.path.join(path, "train"))
    train, dev = split_by_speaker(transcripts, dev_frac=args.dev_frac)
    build_json(train, path, "train")

    print("Preprocessing dev")
    build_json(dev, path, "dev")

    print("Preprocessing test")
    transcripts = load_transcripts(os.path.join(path, "test"))
    build_json(transcripts, path, "test")
