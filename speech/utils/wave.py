from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave

def array_from_wave(file_name):
    wv = wave.open(file_name, 'r')
    audio = np.frombuffer(wv.readframes(-1), dtype=np.int16)
    samp_rate = wv.getframerate()
    wv.close()
    return audio, samp_rate

def wav_duration(file_name):
    wv = wave.open(file_name, 'r')
    nframes = wv.getnframes()
    samp_rate = wv.getframerate()
    duration = nframes / samp_rate
    wv.close()
    return duration
