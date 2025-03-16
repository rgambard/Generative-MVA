import torch
from torchvision import utils
from PIL import Image
import time
def save_image(im,name):
    normalized = torch.clip(im, -20,20)
    utils.save_image(normalized,name)


class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


