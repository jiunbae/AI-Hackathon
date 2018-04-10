import torch

HAS_DATASET = False
IS_ON_NSML = False
DATASET_PATH = ''
GPU_NUM = torch.cuda.device_count()

_save = lambda x: None
_load = lambda x: None
_infer = lambda x: None 

def save(*args, **kwargs):
    global _save
    return _save(*args, **kwargs)

def load(*args, **kwargs):
    global _load
    return _load(*args, **kwargs)

def infer(*args, **kwargs):
    global _infer
    return _infer(*args, **kwargs)

def report(summary, scope, epoch, epoch_total, train__loss, step):
    pass

def bind(save, load, infer):
    global _save, _load, _infer
    _save = save
    _load = load
    _infer = infer
