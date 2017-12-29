import numpy as np
import os
import cPickle as pickle
import speech.utils.ctc_align as ctc_align

model_path = "/afs/cs.stanford.edu/u/awni/scr/speech/examples/timit/models/ctc_ali_save"

def load_probs(idx):
    fn = os.path.join(model_path, "out_{}.npy".format(idx*50))
    grid = np.load(fn)
    return grid

def load_labels():
    fn = os.path.join(model_path, "labels.bin")
    with open(fn, 'r') as fid:
        return pickle.load(fid)


def save_alis(start_idx, end_idx):
    labels = load_labels()
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print "Saved {}.".format(i)
        acts = load_probs(i)
        grid = ctc_align.score(acts, labels)
        grid = grid.T.astype(np.float64)
        grid = np.exp(grid)
        grid = grid / np.sum(grid, axis=0, keepdims=True)
        save_path = os.path.join(model_path, "ali_{}.npy".format(i))
        np.save(save_path, grid.astype(np.float32))

save_alis(0, 500)
