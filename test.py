import argparse
import os
from PIL import Image
import numpy as np
import yaml
import logging 
from tqdm import tqdm 
from sklearn import svm
import lpips


import torch
from torchvision.utils import save_image
from torch.autograd import Variable

from data import define_dataloader
from models import get_generator, Inception
from utils import denormalize
from metrics import ssim, cal_fid, get_feature_images, PSNR, get_feature_images_incep

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default='configs/celeba_inpaintingV3.yaml', help="the config file path")
parser.add_argument("--out_dir", default='./test_out/celeba_0.4-0.6', help="where to save output")
parser.add_argument("--model", type=str, default="./generator_150.pth", help="generator model pass")
parser.add_argument("--image_size", type=int, default=256, help="test image size 256 or 512")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()

os.makedirs(os.path.join(opt.out_dir,"images"), exist_ok=True)

# log
logging.basicConfig(filename=os.path.join(opt.out_dir, "test.log"), level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')
config = yaml.load(open(opt.cfg, encoding="utf-8"), Loader=yaml.Loader)
config["val"]["batch_size"] = 1
config["val"]["num_work"] = 4

# set -1 to use all test set 
config["val"]["config"]["data_len"] = 10000

# Define model and load model checkpoint
generator  = get_generator(config["models"]["generator"])
generator.load_state_dict(torch.load(opt.model))
generator.to(device)
generator.eval()

## you can chose which pretrained model to calcute FID
incep = Inception().to(device)
# fname = "./metrics/inception-2015-12-05.pt"
# incep = torch.jit.load(open(fname, "rb")).to(device)
incep.eval()


# Prepare input
_, test_data = define_dataloader(config)


# Calculate metrices
with torch.no_grad():

    all_ssim = []
    all_real = None
    all_pred = None
    all_psnr = []
    all_lpips = []
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    flag = 0
    
    psnr = PSNR(255)
    for input in tqdm(test_data):
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(img, prior*(1-mask), mask)
        
        all_psnr.append(psnr(img, out).cpu().numpy())
        all_lpips.append(lpips_vgg(img, out).squeeze().cpu())
        # real, pred = get_feature_images(incep, img, out)
        real, pred = get_feature_images_incep(incep, img, out)
        if flag == 0:
            all_real = real
        else:
            all_real = np.concatenate((all_real, real), axis=0)
        if flag == 0:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            
        all_ssim.append(ssim(img, out).cpu().numpy())
        img_mask = img * (1- mask) + mask
        img_grid = denormalize(torch.cat((img_mask, out, img), -1))
        save_image(img_grid, opt.out_dir + "/images/{}.png".format(len(all_ssim)), nrow=1, normalize=False)
        flag += 1
    res = {}
    fid = cal_fid(all_real, all_pred)
    avg_psnr = np.sum(np.array(all_psnr)) / len(all_psnr) 
    avg_ssim = np.sum(np.array(all_ssim)) / len(all_ssim)
    avg_lpips = sum(all_lpips) / len(all_lpips)
    logging.info(f"fid: {fid}, psnr: {avg_psnr}, ssim: {avg_ssim}, lpips: {avg_lpips}")

    
        
