#!/bin/bash

timit_path=$1
python preprocess.py $timit_path
ln -s $timit_path data
