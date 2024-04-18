import argparse
import os
from PIL import Image
import numpy as np
import yaml
import logging 
from tqdm import tqdm 
import urllib
import lpips


import torch
from torchvision.utils import save_image
from torch.autograd import Variable

from data import define_dataloader
from models import get_generator
from utils import denormalize

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default='configs/places2_inpaintingV3.yaml', help="the config file path")
parser.add_argument("--out_dir", default='./test_out/places2_inpainting_0.4-0.6/', help="where to save output")
parser.add_argument("--model", type=str, default="./results/places2_inpainting_0.2-0.4/models/generator_best_0.6.pth", help="generator model pass")
parser.add_argument("--image_size", type=int, default=256, help="test image size 256 or 512")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()

os.makedirs(opt.out_dir, exist_ok=True)
os.makedirs(opt.out_dir+"images", exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')
config = yaml.load(open(opt.cfg, encoding="utf-8"), Loader=yaml.Loader)
config["val"]["batch_size"] = 1
config["val"]["config"]["data_len"] = 2000
# Define model and load model checkpoint
generator  = get_generator(config["models"]["generator"])
generator.load_state_dict(torch.load(opt.model))
generator.to(device)
generator.eval()

# Prepare input
_, test_data = define_dataloader(config)

# Calculate image
with torch.no_grad():

    flag = 1

    for input in tqdm(test_data):
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(img, prior*(1-mask), mask)
        
        img_mask = img * (1- mask) + mask
        img_grid = denormalize(torch.cat((img_mask, out, img), -1))
        save_image(img_grid, opt.out_dir + "images/{}.png".format(flag), nrow=1, normalize=False)
        flag += 1
   