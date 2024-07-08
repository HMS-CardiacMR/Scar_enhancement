from PIL import Image, ImageOps
import os
import numpy as np
from math import log10, sqrt 
import lpips
import torch
from tqdm import tqdm
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils.data_utils import *
import click
import nibabel as nib
import pandas as pd
import math
import matplotlib.pyplot as plt

@click.command()
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
    '--output', 
    '-o', 
    help='Path to where results are stored.', 
    required=True,
)

def main(results_path, splits, output):
    labels_path = "/data1/marta/original_data_labels"
    target_path = "/data1/marta/enhanced_data5"
    results_paths = [
        os.path.join(results_path, "enhanced_scar"),
        os.path.join(results_path, "enhanced_no_scar")
    ]
    max_dir = os.path.join(results_path, output)
    os.makedirs(max_dir, exist_ok=True)
    splits = int(splits)
    prev_imgs = []
    for results_path in results_paths:
        files = os.listdir(results_path)
        for f in tqdm(files):
            if f.endswith(".nii.gz"):
                print(f)
                f_name = f.split(".")[0][:-1]
                print(f_name)
                if f_name in prev_imgs:
                    continue
                cur_img = []
                for s in range(splits):
                    cur_f_name = f_name + str(s) + ".nii.gz"
                    edited_img = nib.load(os.path.join(results_path, cur_f_name)).get_fdata()
                    cur_img.append(edited_img)
                affine = nib.load(os.path.join(results_path, cur_f_name)).affine
                max_img = np.mean(cur_img, axis=0) 
                img = nib.Nifti1Image(max_img, affine)
                nib.save(img, os.path.join(max_dir, f_name + "mean.nii.gz"))   
                plt.axis("off")
                plt.imshow(max_img, cmap="gray")
                plt.savefig(os.path.join(max_dir, f_name + "mean.png"), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                prev_imgs.append(f_name)


if __name__ == '__main__':
    main()
