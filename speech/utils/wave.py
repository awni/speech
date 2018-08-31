from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import soundfile

def array_from_wave(file_name):
    audio, samp_rate = soundfile.read(file_name)
    return audio, samp_rate

def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name)
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration
