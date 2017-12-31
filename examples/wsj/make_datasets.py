import glob
import os

wsj_base = "/deep/group/speech/datasets/wsj-awni"

train = ["wsj1/doc/indices/si_tr_s.ndx",
         "wsj0/doc/indices/train/tr_s_wv1.ndx"]
dev_93 = ["wsj1/doc/indices/h1_p0.ndx"]
eval_92 = ["wsj0/doc/indices/test/nvp/si_et_20.ndx"]

def get_waves(files):
    waves = []
    for f in files:
        flist = os.path.join(wsj_base, f)
        with open(flist, 'r') as fid:
            lines = (l.split(":")[1].strip().strip("/") for l in fid if l[0] != ';')
            lines = (os.path.join(wsj_base, l) for l in lines)
            waves.extend(sorted(lines))
    # TODO replace wv1 with wav
    return waves

def key_from_wv(wv):
    return os.path.basename(wv).split(".")[0]

train_list = get_waves(train)
train_list = filter(lambda x: "wsj0/si_tr_s/401" not in x, train_list)
dev_list = get_waves(dev_93)
test_list = get_waves(eval_92)

# Transcripts
transcripts = []
dots = glob.glob(os.path.join(wsj_base, "wsj0/transcrp/dots/*/*/*.dot"))
dots.extend(glob.glob(os.path.join(wsj_base, "wsj1/trans/wsj1/*/*/*.dot")))
dots.extend(glob.glob(os.path.join(wsj_base, "wsj0/si_et_20/*/*.dot")))
for f in dots:
    with open(f, 'r') as fid:
        transcripts.extend(l.strip() for l in fid)
transcripts = (t.split() for t in transcripts)
transcripts = {t[-1][1:-1] : " ".join(t[:-1]) for t in transcripts}

# All transcripts are here. Now just need to normalize like Kaldi
