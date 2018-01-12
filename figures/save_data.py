import glob
import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import json

char_map = {
    0: u'iy', 1: u'ix', 2: u'aa', 3: u'en', 4: u'ae', 5: u'eh',
    6: u'cl', 7: u'ah', 8: u'ao', 9: u'ih', 10: u'ch', 11: u'ey',
    12: u'aw', 13: u'ay', 14: u'ax', 15: u'er', 16: u'vcl', 17: u'ng',
    18: u'sh', 19: u'th', 20: u'sil', 21: u'el', 22: u'zh', 23: u'w',
    24: u'dh', 25: u'epi', 26: u'ow', 27: u'hh', 28: u'jh', 29: u'dx',
    30: u'b', 31: u'd', 32: u'g', 33: u'f', 34: u'uw', 35: u'm',
    36: u'l', 37: u'n', 38: u'uh', 39: u'p', 40: u's', 41: u'r',
    42: u't', 43: u'oy', 44: u'v', 45: u'y', 46: u'z', 47: u'k',
    48: '</s>', 49: '<s>'}

seq2seq = False
model_path = "/afs/cs.stanford.edu/u/awni/scr/speech/examples/timit/models/{}_ali_save"
if seq2seq:
    model_path = model_path.format("seq2seq")
else:
    model_path = model_path.format("ctc")

with open(os.path.join(model_path, "labels.bin"), 'r') as fid:
    labels = pickle.load(fid)
labels = [char_map[l] for l in labels]

if seq2seq:
    labels = labels[1:]

def errors_from_file():
    f = glob.glob(model_path + "/events*")
    assert len(f) == 1, "Wrong num files"
    log_file = f[0]
    losses = []
    curr_step = 0
    for e, summary in enumerate(tf.train.summary_iterator(log_file)):
        if e == 0:
            continue
        step = summary.step
        if int(step) != curr_step:
            continue
        curr_step += 1
        vals = summary.summary.value.pop()
        loss = vals.simple_value
        losses.append(loss)
    return losses

losses = errors_from_file()
smooth = 80
losses = np.convolve(losses, (1./smooth) * np.ones(smooth), mode="valid")

def get_alis(idx):
    if seq2seq:
        ali_file = "out_{}.npy".format(50 * idx)
    else:
        ali_file = "ali_{}.npy".format(idx)
    file_name = os.path.join(model_path, ali_file)
    alis = np.load(file_name)
    if not seq2seq:
        alis = alis[1::2, :] # remove blanks
    alis = alis.tolist()
    alis = [[round(a, 4) for a in ali] for ali in alis]
    return alis

max_id = 300
alis = [get_alis(i) for i in range(max_id)]
losses = losses[:max_id * 50] # convert to list of rounded floats
losses = [round(l, 2) for l in losses]
all_data = {
    "labels" : labels,
    "losses" : losses,
    "alis" : alis
    }


if seq2seq:
    out_file = '{}_alis.js'.format("seq2seq")
else:
    out_file = '{}_alis.js'.format("ctc")

with open(out_file, 'w') as fid:
    json.dump(all_data, fid)

