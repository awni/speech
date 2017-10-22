#!/bin/bash

# Install sph2pipe
sph_v=sph2pipe_v2.5
wget http://www.openslr.org/resources/3/${sph_v}.tar.gz
tar -xzvf ${sph_v}.tar.gz
cd ${sph_v} && gcc -o sph2pipe *.c -lm
rm ../${sph_v}.tar.gz

# Convert sphere to wave
sph2pipe=$sph_v/sph2pipe

for sph in `cat train_si284.flist test_dev93.flist test_eval92.flist`
do 
    f=`basename -s .wv1 $sph`
    d=`dirname $sph`
    out=$d/${f}.wav
    echo $out
    $sph2pipe -p -f wav -c 1 $sph $out
done
