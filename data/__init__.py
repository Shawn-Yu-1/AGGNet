from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

from .dataset import InpaintDataset, UncroppingDataset, EdgeInpainting

dataset = {"inpainting": InpaintDataset, "uncropping": UncroppingDataset, "edge": EdgeInpainting}

def define_dataloader(opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    
    phase_dataset, val_dataset = define_dataset(opt)

    # create dataloader
    
    dataloader = DataLoader(phase_dataset, batch_size=opt["train"]["batch_size"], shuffle=True, num_workers=opt["train"]["num_work"], drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt["val"]["batch_size"], shuffle=False, num_workers=opt["val"]["num_work"], drop_last=False) 
    
    return dataloader, val_dataloader


def define_dataset(opt):
    ''' loading Dataset() class from given file's name '''
    # dataset_opt = opt['datasets']
    train_dataset = dataset[opt["train"]["name"]](**opt["train"]["config"])
    val_dataset = dataset[opt["val"]["name"]](**opt["val"]["config"])

    return train_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
