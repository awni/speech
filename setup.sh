#!/bin/bash
# Run `source setup.sh` from this directory.
export PYTHONPATH=`pwd`/speech:`pwd`/libs/warp-ctc/pytorch_binding:`pwd`/libs:$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc/build
