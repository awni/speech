#!/bin/bash

# Call this script to preprocess the WSJ datasets.
#
# This script follows the Kaldi setup closely
#    https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj/s5
#
# Usage: 
#    ./preprocess.sh <path_to_wsj> <path_to_save_jsons>
#
# The default behavior is to convert all the *.wv1 SPHERE files
# in the wsj corpus to 'wav' format. Thus you will need write access
# to <path_to_wsj>.
#
# Upon completion three files will be created and saved in
# <path_to_save_jsons>:
#    - train_si284.json (37318 utts)
#    - dev_93.json (503 utts)
#    - eval_92.json (333 utts)

# Path where the dataset is stored.
wsj_base=$1

# Path to save dataset jsons.
save_path=$2

# Install sph2pipe
sph_v=sph2pipe_v2.5
wget http://www.openslr.org/resources/3/${sph_v}.tar.gz
tar -xzvf ${sph_v}.tar.gz
cd ${sph_v} && gcc -o sph2pipe *.c -lm
cd ..
rm ${sph_v}.tar.gz

python preprocess.py $1 $2 --convert
rm -rf $sph_v
