from __future__ import print_function

import argparse
import os
import urllib.request
import tarfile

EXT = ".tar.gz"
FILES = ["raw-metadata", "train-clean-100", "dev-clean"]
BASE_URL = "http://www.openslr.org/resources/12/"

def download_and_extract(in_file, out_dir):
    in_file = in_file + EXT
    file_url = os.path.join(BASE_URL, in_file)
    out_file = os.path.join(out_dir, in_file)

    # Download and extract zip file.
    urllib.request.urlretrieve(file_url, filename=out_file)
    with tarfile.open(out_file) as tf:
        tf.extractall(path=out_dir)

    # Remove zip file after use
    os.remove(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download librispeech dataset.")

    parser.add_argument("output_directory",
        help="The dataset is saved in <output_directory>/LibriSpeech.")
    args = parser.parse_args()

    for f in FILES:
        print("Downloading {}".format(f))
        download_and_extract(f, args.output_directory)
