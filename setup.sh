#!/bin/bash
# Run `source setup.sh` from this directory.
export PYTHONPATH=`pwd`/..:`pwd`/libs/warp-ctc/pytorch_binding:$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc/build
