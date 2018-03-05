from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

FFMPEG = "ffmpeg"
AVCONV = "avconv"

def check_install(*args):
    try:
        subprocess.check_output(args,
                    stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False

def check_avconv():
    """
    Check if avconv is installed.
    """
    return check_install(AVCONV, "-version")

def check_ffmpeg():
    """
    Check if ffmpeg is installed.
    """
    return check_install(FFMPEG, "-version")


USE_AVCONV = check_avconv()
USE_FFMPEG = check_ffmpeg()
if not (USE_AVCONV or USE_FFMPEG):
    raise OSError(("Must have avconv or ffmpeg "
                   "installed to use conversion functions."))
USE_AVCONV = not USE_FFMPEG

def to_wave(audio_file, wave_file, use_avconv=USE_AVCONV):
    """
    Convert audio file to wave format.
    """
    prog = AVCONV if use_avconv else FFMPEG
    args = [prog, "-y", "-i", audio_file, "-f", "wav", wave_file]
    subprocess.check_output(args, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    print("Use avconv", USE_AVCONV)

