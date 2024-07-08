"""
    File to generate scar enhanced data using tissue specific gamma correction.
    To run the file run a command:
    python generate_data.py -o OUTPUT_FOLDER
"""

import os
import click
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils.data_utils import *
import pandas as pd
from tqdm import tqdm
import nibabel as nib

ROOT_DIR = "/mnt/alp/Users/Marta"

@click.command()
@click.option(
    '--output_folder', 
    '-o', 
    help='Path where you want to store the gamma enhanced images.', 
    required=True
)
@click.option(
    '--train_csv_path', 
    '-t', 
    help='Path to the csv file including information on train data.', 
    required=False,
    default="splits/train.csv"
)
@click.option(
    '--val_csv_path', 
    '-v', 
    help='Path to the csv file including information on train data.', 
    required=False,
    default="splits/val.csv"
)
@click.option(
    '--test_csv_path', 
    '-s', 
    help='Path to the csv file including information on train data.', 
    required=False,
    default="splits/test.csv"
)

def main(output_folder, train_csv_path, val_csv_path, test_csv_path):
    os.makedirs(output_folder, exist_ok=True)
    blood_gamma = float(blood_gamma)
    myo_gamma = float(myo_gamma)
    scar_gamma = float(scar_gamma)
    train_csv = pd.read_csv(train_csv_path)
    train_data = train_csv['img_path_original'].to_numpy()
    train_label = train_csv['label_path'].to_numpy()
    val_csv = pd.read_csv(val_csv_path)
    val_data = val_csv['img_path_original'].to_numpy()
    val_label = val_csv['label_path'].to_numpy()
    test_csv = pd.read_csv(test_csv_path)
    test_data = test_csv['img_path_original'].to_numpy()
    test_label = test_csv['label_path'].to_numpy()
    make_data(blood_gamma, myo_gamma, scar_gamma, train_data, train_label, output_folder)
    make_data(blood_gamma, myo_gamma, scar_gamma, val_data, val_label, output_folder)
    make_data(blood_gamma, myo_gamma, scar_gamma, test_data, test_label, output_folder)

def make_data(blood_gamma, myo_gamma, scar_gamma, data, labels, output_folder):
    all_data = []
    data_dir = "/data1/marta"
    for i, mr in enumerate(tqdm(data)):
        lbl = labels[i]
        all_data.append({
            "mr" : nib.load(os.path.join(data_dir, mr)).get_fdata(),
            "label" : nib.load(os.path.join(data_dir, lbl)).get_fdata(),
            "mr_file" : mr,
            "label_file" : lbl,
            "affine" : nib.load(os.path.join(data_dir, mr)).affine
        })
    for d in tqdm(all_data):
        mr_edited = np.zeros_like(d['mr'])
        for j in range(3):
            slice,b,m,s = tissue_specific_gamma_correction(d['mr'][:,:,j], d['label'][:,:,j], scar_gamma, myo_gamma, blood_gamma)  
            mr_edited[:,:,j] = slice
        img = nib.Nifti1Image(mr_edited, d['affine'])
        nib.save(img, os.path.join(output_folder, d['mr_file'].split("/")[-1]))    


if __name__ == '__main__':
    main()