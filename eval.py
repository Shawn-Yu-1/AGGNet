from metrics import ssim, cal_fid, get_feature_images, PSNR, get_feature_images_incep
import torch
import numpy as np
from tqdm import tqdm


def calcu_eval(generator, percep, dataset, device, is_fid=True, is_psnr=True, is_ssim=False):
    
    if is_ssim:
        all_ssim = []
    all_real = []
    all_pred = []
    all_psnr = []
    
    psnr = PSNR(255)
    
    for input in tqdm(dataset):
        
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(img, prior*(1-mask), mask)
        
        all_psnr.append(psnr(input["image"], out.cpu()).numpy())
        
        # real, pred = get_feature_images(percep, input["image"], out.cpu())
        real, pred = get_feature_images_incep(percep, input["image"], out.cpu())
        if all_real == []:
            all_real = real
        else:
            all_real = np.concatenate((all_real, real), axis=0)
        if all_pred == []:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            
        if is_ssim:
            all_ssim.append(ssim(input["image"], out.cpu()).numpy())
            
    res = {}
    res["fid"] = cal_fid(all_real, all_pred)
    res["psnr"] = np.sum(np.array(all_psnr)) / len(all_psnr)
    if is_ssim:
        res["ssim"] = np.sum(np.array(all_ssim)) / len(all_ssim)
    return res

def calcu_eval_cuda(generator, percep, dataset, device, is_fid=True, is_psnr=True, is_ssim=False):
    
    if is_ssim:
        all_ssim = []
    all_real = []
    all_pred = []
    all_psnr = []
    
    psnr = PSNR(255)
    percep = percep.to(device)
    
    for input in tqdm(dataset):
        
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(img, prior, mask)
        
        all_psnr.append(psnr(input["image"], out).numpy())
        
        real, pred = get_feature_images(percep, input["image"], out)
        if all_real == []:
            all_real = real
        else:
            all_real = np.concatenate((all_real, real), axis=0)
        if all_pred == []:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            
        if is_ssim:
            all_ssim.append(ssim(input["image"], out).numpy())
            
    res = {}
    res["fid"] = cal_fid(all_real, all_pred)
    res["psnr"] = np.sum(np.array(all_psnr)) / len(all_psnr)
    if is_ssim:
        res["ssim"] = np.sum(np.array(all_ssim)) / len(all_ssim)
    return res


def calcu_eval_edge(generator, percep, dataset, device, is_fid=True, is_psnr=True, is_ssim=False):
    
    if is_ssim:
        all_ssim = []
    all_psnr = []
    
    psnr = PSNR(255)
    
    for input in tqdm(dataset):
        
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(img, prior, mask)
        
        all_psnr.append(psnr(input["prior"], out.cpu()).numpy())
        
            
        if is_ssim:
            all_ssim.append(ssim(input["prior"], out.cpu()).numpy())
            
    res = {}
    res["psnr"] = np.sum(np.array(all_psnr)) / len(all_psnr)
    if is_ssim:
        res["ssim"] = np.sum(np.array(all_ssim)) / len(all_ssim)
    return res

def calcu_eval_stage(generator, stage1, percep, dataset, device, is_fid=True, is_psnr=True, is_ssim=False):
    
    if is_ssim:
        all_ssim = []
    all_real = []
    all_pred = []
    all_psnr = []
    
    psnr = PSNR(255)
    
    for input in tqdm(dataset):
        
        img = input["image"].to(device)
        prior = input["prior"].to(device)
        mask = input["mask"].to(device)
        
        out = generator(stage1(img, prior*(1-mask), mask), mask)
        
        all_psnr.append(psnr(input["image"], out.cpu()).numpy())
        
        # real, pred = get_feature_images(percep, input["image"], out.cpu())
        real, pred = get_feature_images_incep(percep, input["image"], out.cpu())
        if all_real == []:
            all_real = real
        else:
            all_real = np.concatenate((all_real, real), axis=0)
        if all_pred == []:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)
            
        if is_ssim:
            all_ssim.append(ssim(input["image"], out.cpu()).numpy())
            
    res = {}
    res["fid"] = cal_fid(all_real, all_pred)
    res["psnr"] = np.sum(np.array(all_psnr)) / len(all_psnr)
    if is_ssim:
        res["ssim"] = np.sum(np.array(all_ssim)) / len(all_ssim)
    return res