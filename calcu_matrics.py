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

from models import get_generator, Inception
from utils import denormalize
from metrics import ssim, cal_fid, get_feature_images, PSNR, get_feature_images_incep

out_dir = "./compare/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
model = "madf"
data = "celeba"
mask_rate = "0.4-0.6"
# log
logging.basicConfig(filename=os.path.join(out_dir, "test.log"), level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda:{}'.format(0))
else:
    device = torch.device('cpu')
    
## you can chose which pretrained model to calcute FID
incep = Inception().to(device)
# fname = "./metrics/inception-2015-12-05.pt"
# incep = torch.jit.load(open(fname, "rb")).to(device)
incep.eval()

# load images
data_root = "/data/dataset/CelebA-HQ/val/"
pre_root = "./test_image/madf/celeba/0.4-0.6/"

real_imgs = []
for root, _, files in os.walk(data_root):
    for file in files:
        real_imgs.append(os.path.join(root, file))
        
# Calculate image

all_ssim = []
all_real = None
all_pred = None
all_psnr = []
all_lpips = []
flag = 0
real_imgs = sorted(real_imgs)

lpips_vgg = lpips.LPIPS(net="vgg").to(device)

psnr = PSNR(255)
for input in tqdm(real_imgs):
    img = np.array(Image.open(input).convert("RGB").resize((256,256))).transpose(2,0,1)
    pre_name  = input.split("/")[-1].replace(".jpg", ".png")
    pre_img = np.array(Image.open(os.path.join(pre_root, pre_name)).convert("RGB")).transpose(2,0,1)
    img = torch.from_numpy(np.array([(img / 127.5) - 1], dtype=np.float32))
    pre_img = torch.from_numpy(np.array([(pre_img / 127.5) - 1], dtype=np.float32))
    
    # all_psnr.append(psnr(img, pre_img))
    
    with torch.no_grad():
        all_lpips.append(lpips_vgg(img.to(device), pre_img.to(device)).squeeze().cpu())
#     print(all_lpips)
#     real, pred = get_feature_images_incep(incep, img.to(device), pre_img.to(device))
#     # real, pred = get_feature_images_incep(incep, img, out)
#     if flag == 0:
#         all_real = real
#     else:
#         all_real = np.concatenate((all_real, real), axis=0)
#     if flag == 0:
#         all_pred = pred
#     else:
#         all_pred = np.concatenate((all_pred, pred), axis=0)
        
#     all_ssim.append(ssim(img, pre_img).cpu().numpy())
#     flag += 1
# fid = cal_fid(all_real, all_pred)
# avg_psnr = np.sum(np.array(all_psnr)) / len(all_psnr) 
# avg_ssim = np.sum(np.array(all_ssim)) / len(all_ssim)
avg_lpips = sum(all_lpips) / len(all_lpips)
# logging.info(f"model: {model} dataset: {data} mask-rate: {mask_rate} fid: {fid}, psnr: {avg_psnr}, ssim: {avg_ssim}, lpips: {avg_lpips}")
logging.info(f"model: {model} dataset: {data} mask-rate: {mask_rate} lpips: {avg_lpips}")

        
