import numpy as np
import os
import cPickle as pickle
import transducer.ref_transduce as rt

model_path = "/afs/cs.stanford.edu/u/awni/scr/speech/examples/timit/models/trans_ali_save"

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
        blank = acts.shape[2]-1
        alphas, ll_forward = rt.forward_pass(acts, labels, blank)
        betas, ll_backward = rt.backward_pass(acts, labels, blank)
        grid = np.exp(alphas + betas)
        grid = grid.T
        grid = grid / np.sum(grid, axis=0, keepdims=True)
        grid = grid.astype(np.float32)
        grid = grid[1:,:] # Get rid of first null token
        save_path = os.path.join(model_path, "ali_{}.npy".format(i))
        np.save(save_path, grid.astype(np.float32))

save_alis(100, 300)
