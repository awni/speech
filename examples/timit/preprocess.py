from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os
import tqdm

from speech.utils import data_helpers
from speech.utils import wave

SETS = ["train", "test"]
WAV_EXT = "wv" # using wv since NIST took wav

def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.phn")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = [l.split()[-1] for l in lines]
            data[f] = phonemes
    return data

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.wav",
            new_ext=WAV_EXT,
            use_avconv=False)

def build_json(path):
    transcripts = load_transcripts(path)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path) + os.path.extsep + "json"
    with open(os.path.join(dirname, basename), 'w') as fid:
        for k, t in tqdm.tqdm(transcripts.items()):
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
        help="The dataset is saved in <output_directory>/LDC93S1-TIMIT")
    args = parser.parse_args()

    path = os.path.join(args.output_directory, "LDC93S1-TIMIT/timit")

    print("Converting files from NIST to standard wave format...")
    convert_to_wav(path)
    for d in SETS:
        print("Preprocessing {}".format(d))
        prefix = os.path.join(path, d)
        build_json(prefix)
