import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import random

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, prior_root, mask_config={}, data_len=-1, image_size=[256, 256], is_train=False):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = sorted(imgs)[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_p = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.data_root = data_root
        self.prior_root = prior_root
        self.mask_config = mask_config
        self.mask_root = self.mask_config["root"]
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.is_train = is_train

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = Image.open(path)
        # prior_path = path.replace(self.data_root, self.prior_root)
        prior_path = os.path.join(self.prior_root,path.split("/")[-1])
        # img_gray = self.tfs_p(Image.open(path).convert("L"))
        prior = Image.open(prior_path).convert("L")
        
        # data augmentation
        if self.is_train and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            prior = prior.transpose(Image.FLIP_LEFT_RIGHT)
        
        img = self.tfs(img)
        prior = self.tfs_p(prior)
        mask = self.get_mask(index)
        
        ret['image'] = img
        ret['prior'] = prior
        ret['mask'] = mask
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index=0):
        if self.mask_mode == 'manual':
            mask = torch.from_numpy(np.array([np.asarray(Image.open(self.mask_root+f"{index % 400}.png"))]))
            return mask
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, prior_root, mask_config={}, data_len=-1, image_size=[256, 256]):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_p = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.prior = prior_root
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        prior_path = os.path.join(self.prior, path.split("/")[-1].replace(".jpg", ".png"))
        img = self.tfs(Image.open(path))
        prior = self.tfs_p(Image.open(prior_path).convert("L"))
        mask = self.get_mask()
        # cond_image = img*(1. - mask) + mask*torch.randn_like(img)

        ret['image'] = img
        ret['prior'] = prior
        ret['mask'] = mask
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class EdgeInpainting(data.Dataset):
    def __init__(self, data_root, prior_root, mask_config={},data_len=-1, image_size=[224, 224]):
        self.data_root = data_root
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = sorted(imgs)[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_p = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.prior_root = prior_root
        self.image_size = image_size
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.mask_root = self.mask_config["root"]
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        
        path = self.imgs[index]
        prior_path = os.path.join(self.prior_root, path.split("/")[-1])
        img = self.tfs_p(Image.open(path).convert("L"))
        prior_rgb = self.tfs_p(Image.open(prior_path).convert("RGB"))
        prior = self.tfs_p(Image.open(prior_path).convert("L"))
        mask = self.get_mask(index)

        ret['image'] = img
        ret['prior_rgb'] = prior_rgb
        ret['prior'] = prior
        ret['mask'] = mask
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index=0):
        if self.mask_mode == 'manual':
            mask = torch.from_numpy(np.array([np.asarray(Image.open(self.mask_root+f"{index % 400}.png"))]))
            return mask
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
