from __future__ import print_function

import subprocess

FFMPEG = "ffmpeg"
AVCONV = "avconv"

def check_install(*args):
    try:
        out = subprocess.check_output(args,
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

def flac_to_wave(flac_file, wave_file, use_avconv=True):
    prog = AVCONV if use_avconv else FFMPEG
    args = [prog, "-i", input_file, output_file]
    subprocess.check_output(args)

if __name__ == "__main__":
    check_avconv()
    check_ffmpeg()
