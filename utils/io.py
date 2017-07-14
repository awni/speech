
import os
import torch

def save(model, path):
    torch.save(model, os.path.join(path, "model"))
