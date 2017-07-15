#!/bin/bash

timit_path=$1/LDC93S1-TIMIT/
python preprocess.py $timit_path
ln -s $timit_path data
