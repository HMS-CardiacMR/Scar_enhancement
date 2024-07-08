from PIL import Image, ImageOps
from skimage.metrics import structural_similarity
import os
import numpy as np
from math import log10, sqrt 
import lpips
import torch
from tqdm import tqdm
import os
from data_utils import *
import click
import nibabel as nib
import pandas as pd
import math

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='vgg').to(device)

def normalize_image2(image_tensor):
    # Normalize image to range [-1, 1]
    return (image_tensor / 255.0 - 0.5) * 2.0

def calculate_lpips_score(original, edited):
    # Load and normalize the images
    image1_tensor = torch.tensor(original).to(device)
    image2_tensor = torch.tensor(edited).to(device)

    image1_normalized = normalize_image2(image1_tensor)
    image2_normalized = normalize_image2(image2_tensor)

    # Calculate the LPIPS score
    lpips_score = loss_fn(image1_normalized, image2_normalized)

    return lpips_score.item()

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

def cnr(tissue1, tissue2):
    if len(tissue2) == 0 or len(tissue1) == 0:
        return 0
    return abs((np.mean(tissue1)-np.mean(tissue2))/np.std(tissue2))

@click.command()
@click.option(
    '--roi', 
    '-r', 
    type=bool,
    help='Whether the metrics are for the ROI.', 
    required=True,
)
@click.option(
    '--results_path', 
    '-p', 
    help='Path to where results are stored.', 
    required=True,
)
@click.option(
    '--splits', 
    '-s', 
    help='Path to where results are stored.', 
    required=True,
)
@click.option(
    '--max', 
    '-m', 
    help='Path to where results are stored.', 
    required=True,
    type=bool
)

def main(roi, results_path, splits, max):
    labels_path = "/data1/marta/original_data_labels"
    target_path = "/data1/marta/enhanced_data5"
    if max: 
        results_paths = [
            os.path.join(results_path, "mean_img5")
        ]
    else:
        results_paths = [
            os.path.join(results_path, "enhanced_scar"),
            os.path.join(results_path, "enhanced_no_scar")
        ]
    splits = int(splits)

    for s in range(splits):
        ssim = []
        lpips_metric = []
        psnr = []
        scar_myo = []
        myos = []
        scar_blood = []
        scars = []
        blood_myo = []
        bloods = []
        for results_path in results_paths:
            files = os.listdir(results_path)
            for f in files:
                if max:
                    if ".nii.gz" in f:
                        target_f = f.split("_")[0] + "_image.nii.gz" # + f.split("_")[2] + ".png" 
                        # target_img = np.asarray(ImageOps.grayscale(Image.open(os.path.join(target_path, target_f))))
                        # edited_img = np.asarray(ImageOps.grayscale(Image.open(os.path.join(results_path, f))))
                        target_img = normalize_image(nib.load(os.path.join(target_path, target_f)).get_fdata()[:,:,int(f[17:18])], 0, 255)
                        edited_img = normalize_image(nib.load(os.path.join(results_path, f)).get_fdata(), 0, 255)
                        label_f_name = f[:16] + "_label.nii.gz"
                        label = nib.load(os.path.join(labels_path, label_f_name)).get_fdata()[:,:,int(f[17:18])]
                        blood, myo, scar = mri_masked(edited_img,label)
                        bloods.append(np.mean(blood))
                        myos.append(np.mean(myo))
                        scars.append(np.mean(scar))
                        if math.isinf(cnr(scar, blood)):
                            scar_blood.append(0)
                        else:
                            scar_blood.append(abs(cnr(scar, blood)))
                        scar_myo.append(abs(cnr(scar, myo)))
                        blood_myo.append(abs(cnr(blood,myo)))
                        if roi:
                            target_img = extract_roi(target_img, label)
                            edited_img = extract_roi(edited_img, label)
                        ssim.append(structural_similarity(target_img, edited_img, data_range=255))
                        lpips_metric.append(calculate_lpips_score(target_img.astype(np.float32), edited_img.astype(np.float32)))
                        psnr.append(PSNR(target_img, edited_img))
                else:
                    if str(s) + ".nii.gz" in f:
                        target_f = f.split("_")[0] + "_image.nii.gz" # + f.split("_")[2] + ".png" 
                        # target_img = np.asarray(ImageOps.grayscale(Image.open(os.path.join(target_path, target_f))))
                        # edited_img = np.asarray(ImageOps.grayscale(Image.open(os.path.join(results_path, f))))
                        target_img = normalize_image(nib.load(os.path.join(target_path, target_f)).get_fdata()[:,:,int(f[17:18])], 0, 255)
                        edited_img = normalize_image(nib.load(os.path.join(results_path, f)).get_fdata(), 0, 255)
                        label_f_name = f[:16] + "_label.nii.gz"
                        label = nib.load(os.path.join(labels_path, label_f_name)).get_fdata()[:,:,int(f[17:18])]
                        blood, myo, scar = mri_masked(edited_img,label)
                        bloods.append(np.mean(blood))
                        myos.append(np.mean(myo))
                        scars.append(np.mean(scar))
                        if math.isinf(cnr(scar, blood)):
                            scar_blood.append(0)
                        else:
                            scar_blood.append(abs(cnr(scar, blood)))
                        scar_myo.append(abs(cnr(scar, myo)))
                        blood_myo.append(abs(cnr(blood,myo)))
                        if roi:
                            target_img = extract_roi(target_img, label)
                            edited_img = extract_roi(edited_img, label)
                        ssim.append(structural_similarity(target_img, edited_img, data_range=255))
                        lpips_metric.append(calculate_lpips_score(target_img.astype(np.float32), edited_img.astype(np.float32)))
                        psnr.append(PSNR(target_img, edited_img))

        cur_dir = results_path.split('/')[-1]

        df = pd.DataFrame()
        df['lpips'] = lpips_metric
        df['ssim'] = ssim
        df['psnr'] = psnr
        df['scar_blood'] = scar_blood
        df['scar_myo'] = scar_myo
        df['blood_myo'] = blood_myo
        df['blood'] = bloods
        df['myo'] =  myos
        df['scar'] = scars

        df.to_csv(results_path + str(s) + ".csv")

        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips_metric)
        mean_psnr = np.mean(psnr)

        print("RESULTS FROM ", cur_dir, str(s))
        print(f'Mean SSIM: {mean_ssim:.4f}')
        print(f'Mean LPIPS: {mean_lpips:.4f}')
        print(f'Mean PSNR: {mean_psnr:.4f}')



if __name__ == '__main__':
    main()
