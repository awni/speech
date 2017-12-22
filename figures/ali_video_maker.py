import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import cPickle as pickle

fig, ax = plt.subplots(figsize=(16, 5))

char_map = {0: u'iy', 1: u'ix', 2: u'aa', 3: u'en', 4: u'ae', 5: u'eh', 6: u'cl', 7: u'ah', 8: u'ao', 9: u'ih', 10: u'ch', 11: u'ey', 12: u'aw', 13: u'ay', 14: u'ax', 15: u'er', 16: u'vcl', 17: u'ng', 18: u'sh', 19: u'th', 20: u'sil', 21: u'el', 22: u'zh', 23: u'w', 24: u'dh', 25: u'epi', 26: u'ow', 27: u'hh', 28: u'jh', 29: u'dx', 30: u'b', 31: u'd', 32: u'g', 33: u'f', 34: u'uw', 35: u'm', 36: u'l', 37: u'n', 38: u'uh', 39: u'p', 40: u's', 41: u'r', 42: u't', 43: u'oy', 44: u'v', 45: u'y', 46: u'z', 47: u'k', 48: '</s>', 49: '<s>'}

seq2seq = True

model_path = "/afs/cs.stanford.edu/u/awni/scr/speech/examples/timit/models/seq2seq_ali_save"
with open(os.path.join(model_path, "labels.bin"), 'r') as fid:
    labels = pickle.load(fid)
labels = [char_map[l] for l in labels]

if seq2seq:
    labels = labels[1:]

def get_plot(idx):
    file_name = os.path.join(model_path, "out_{}.npy".format(50 * idx))
    alis = np.load(file_name)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    p = ax.imshow(alis)
    ax.set_xlabel("Inputs")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", which="both", right="off", left="off")
    ax.set_ylabel("Outputs")
    return [p]

def load_alis(max_id=500):
    ims = []
    for i in range(0, max_id):
        ims.append(get_plot(i))
    return ims

alis = load_alis()
im_ani = animation.ArtistAnimation(fig, alis, interval=100,
            repeat=False, blit=True)
im_ani.save('test_ali_vid.mp4', metadata={'artist':'Awni Hannun'})
