
import os
import cPickle as pickle
import torch

MODEL = "model"
PREPROC = "preproc.pyc"

def save(model, preproc, path):
    torch.save(model, os.path.join(path, MODEL))
    with open(os.path.join(path, PREPROC), 'w') as fid:
        pickle.dump(preproc, fid)

def load(path):
    model = torch.load(os.path.join(path, MODEL))
    with open(os.path.join(path, PREPROC), 'r') as fid:
        preproc = pickle.load(fid)
    return model, preproc


