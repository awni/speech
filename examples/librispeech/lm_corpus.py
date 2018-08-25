from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import gzip
import urllib

import subprocess

EXT = '.txt.gz'
FILE = 'librispeech-lm-norm'
BASE_URL = 'http://www.openslr.org/resources/11/'

LMPLZ = 'lmplz'
BUILD_BINARY = 'build_binary'
LM_CORPUS = 'librispeech-lm-5gram-prune01111.trie.klm'


def build_lm_corpus(in_file, out_dir):
    in_file = in_file + EXT
    file_url = os.path.join(BASE_URL, in_file)
    out_file = os.path.join(out_dir, in_file)

    # Download and extract zip file.
    urllib.urlretrieve(file_url, filename=out_file)

    with open(FILE + '.txt', 'w') as normalized:
        with io.TextIOWrapper(io.BufferedReader(gzip.open(out_file)), encoding='utf8') as raw:
            for line in raw:
                normalized.write(line.lower())

    os.remove(out_file)

    args = [LMPLZ,
            '--order', '5',
            '--text', FILE + '.txt',
            '--arpa', FILE + '.arpa',
            '--prune', '0', '1', '1', '1', '1']
    subprocess.check_output(args, stderr=subprocess.STDOUT)

    args = [BUILD_BINARY,
            '-a', '22',
            '-q', '8',
            '-b', '8',
            'trie', FILE + '.arpa',
            LM_CORPUS]
    subprocess.check_output(args, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download librispeech language model corpus.')
    parser.add_argument('output_directory', help='The corpus is saved in <output_directory>.')
    args = parser.parse_args()

    print('Building language model: {}'.format(FILE))
    build_lm_corpus(FILE, args.output_directory)
