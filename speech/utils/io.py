
import os
import cPickle as pickle
import torch

MODEL = "model"
PREPROC = "preproc.pyc"

def get_names(path, tag):
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + MODEL)
    preproc = os.path.join(path, tag + PREPROC)
    return model, preproc

def save(model, preproc, path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    torch.save(model, model_n)
    with open(preproc_n, 'w') as fid:
        pickle.dump(preproc, fid)

def load(path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    model = torch.load(model_n) if \
        torch.cuda.is_available() else \
        torch.load(model_n, map_location=lambda storage, loc: storage)

    with open(preproc_n, 'rb') as fid:
        preproc = pickle.load(fid)
    return model, preproc


