import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def gray2rgb(tensor: torch.TensorType):
    """_summary_

    Args:
        tensor : B1HW tensor to B3HW tensor
    """
    tag = False
    imgs = tensor.detach().cpu().numpy()
    for i in range(imgs.shape[0]):
        gray = imgs[i, :, :, :].squeeze()
        gray = (gray + 1) * 127.5
        img = Image.fromarray(gray).convert("RGB")
        img = np.asarray(img).transpose(2, 0, 1) / 127.5 - 1
        if tag == False:
            tag = True
            out = np.array([img])
        else:
            out = np.concatenate((out, [img]), axis=0)         
    return torch.from_numpy(out)
        