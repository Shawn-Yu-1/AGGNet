import argparse
from datetime import datetime
import os
import yaml
from torch import nn
import logging 
import torch
from PIL import Image
import numpy as np

from torchvision.utils import save_image

from data import define_dataloader
from models import get_generator, Discriminator512, Inception, GatedAttentionNetV2
from torchsummary import summary
from utils import denormalize, set_seed
from eval import calcu_eval
from models.loss import PerceptualLoss

t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="./configs/places2_inpainting.yaml", help="the config files")
parser.add_argument("--resume", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--lambda_adv", type=float, default=0.1, help="adversarial loss weight")
parser.add_argument("--lambda_l1", type=float, default=5, help="l1 loss weight")
parser.add_argument("--out_dir", default='', help="where to store the output")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()


config = yaml.load(open(opt.cfg, encoding="utf-8"), Loader=yaml.Loader)
if opt.out_dir != "":
    config["out_dir"] = opt.out_dir

# set global random seeds
set_seed(config["seed"])

# set sample and models save dir
img_dir = os.path.join(config["out_dir"], "images")
model_dir = os.path.join(config["out_dir"], "models")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(config["out_dir"], "train.log"), level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

img_size = opt.hr_shape
# Initialize generator and discriminator
generator = get_generator(config["models"]["generator"]).to(device)
discriminator = Discriminator512(4).to(device)
# fname = "./metrics/inception-2015-12-05.pt"
# incep = torch.jit.load(open(fname, "rb")).cpu()
incep = Inception().cpu()
incep.eval()

# Summary of the networks
summary(generator, [(3, img_size, img_size), (1, img_size, img_size), (1, img_size, img_size)])
summary(discriminator, [(3, img_size, img_size), (1, img_size, img_size)])
# summary(ie, (3, hr_shape, hr_shape))

# Set feature extractor to inference mode
# incep.eval()

# Losses
vgg_loss = PerceptualLoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)


if config["iter"] != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(model_dir+"/generator_%d.pth" % config["iter"]))
    discriminator.load_state_dict(torch.load(model_dir+"/discriminator_%d.pth" % config["iter"]))
    print("models loaded")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["optim"]["lr_g"], betas=(config["optim"]["b1"], config["optim"]["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["optim"]["lr_d"], betas=(config["optim"]["b1"], config["optim"]["b2"]))

# load dataset
train_data, val_data = define_dataloader(config)

# ----------
#  Training
# ----------

# init best psnr metric
best_psnr = 0

iter = config["iter"]
for epoch in range(config["resume"], config["epoch"]):
    D_loss = 0
    G_loss = 0
    percep = 0
    adv = 0
    pixel = 0
    style = 0
    t = datetime.now()
    for i, imgs in enumerate(train_data):
        # Configure model input
        img = imgs["image"].to(device)
        prior = imgs["prior"].to(device)
        mask = imgs["mask"].to(device)
        img_mask = img * (1-mask) + torch.rand(img.shape).to(device) * mask
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a extended image from input
        gen_img = generator(img, prior, mask)
        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_img, img)
        pred_fake = discriminator(gen_img, mask)
        # Adversarial loss (relativistic average GAN)
        loss_GAN = -pred_fake.mean()
        loss_percep, loss_style = vgg_loss(img, gen_img)
        # Total generator loss
        loss_G = opt.lambda_adv * loss_GAN + loss_pixel * opt.lambda_l1 + loss_percep * 0.5 + loss_style * 200

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        
        gen_img = generator(img, prior, mask)
        pred_real = discriminator(img, mask)
        pred_fake = discriminator(gen_img, mask)

        # Total loss
        loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        D_loss += loss_D.item()
        G_loss += loss_G.item()
        adv += loss_GAN.item()
        pixel += loss_pixel.item()
        percep += loss_percep.item()
        style += loss_style.item()
        if i % 1000 == 0:
            print(f"Epoch:{epoch+1}/{opt.epochs}, iter: {i}, t/iter: {(datetime.now() - t_start) / (i+1)}")
        iter += 1
        if iter % 2000 == 0:
            avg_D_loss = D_loss / 2000
            avg_G_loss = G_loss / 2000
            avg_adv_loss = adv / 2000
            avg_pixel_loss = pixel / 2000
            avg_percep_loss = percep / 2000
            avg_style_loss = style / 2000

            logging.info(
                'Epoch:{1}/{2} iter:{9} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} percep_loss:{7} style_loss:{8} time:{0}'.format(
                    datetime.now() - t, epoch + 1, opt.epochs, avg_D_loss,
                    avg_G_loss, avg_adv_loss, avg_pixel_loss, avg_percep_loss, avg_style_loss, iter))
            
            D_loss = 0
            G_loss = 0
            percep = 0
            adv = 0
            pixel = 0
            style = 0
            
        if iter % config["eval_iter"] == 0:
            generator.eval()
            with torch.no_grad():
                metrice = calcu_eval(generator, incep, val_data, device, is_ssim=True)
            logging.info(f"val dataset contain {len(val_data)} images")
            logging.info("epoch: {}, iter: {}, fid score: {}, psnr score: {}, ssim: {}".format(epoch+1, iter, metrice["fid"], metrice["psnr"], metrice["ssim"]))
            # save the best model
            if metrice["psnr"] > best_psnr:
                best_psnr = metrice["psnr"]
                torch.save(generator.state_dict(), model_dir + "/generator_best.pth")
            generator.train()
            
        if iter % config["sample_iter"] == 0:
            # Save example results
            img_grid = denormalize(torch.cat((img_mask, gen_img, img), -1))
            save_image(img_grid, img_dir + "/iter-{}.png".format(iter), nrow=1, normalize=False)
            
        if iter % config["save_model"] == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), model_dir + "/generator_{}.pth".format(iter))
            torch.save(discriminator.state_dict(), model_dir + "/discriminator_{}.pth".format(iter))
            logging.info(f"iter {iter} model saved!")

