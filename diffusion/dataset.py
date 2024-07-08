import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import os
from torchvision import transforms
import warnings
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils.data_utils import *

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class MRDataset(Dataset):
    def __init__(self, csv_path, mode = "normal"):
        self.csv_path = csv_path
        self.csv_file = pd.read_csv(self.csv_path)
        self.data = self.csv_file.to_dict(orient='records')
        self.mode = mode
        self.data_dir = "/data1/marta"
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        current_slice = self.data[index]
        img = nib.load(os.path.join(self.data_dir, current_slice['img_path_original'])).get_fdata()[:,:,current_slice['slice']]
        label = nib.load(os.path.join(self.data_dir, current_slice['label_path'])).get_fdata()[:,:,current_slice['slice']]
        # image_mean = np.mean(img)
        # image_std = np.std(img)
        # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        # img = np.clip(img, img_range[0], img_range[1])
        # img = img / (img_range[1] - img_range[0])
        img = normalize_image(img, -1, 1)
        if self.mode == "enhancement":
            enhanced = nib.load(os.path.join(self.data_dir, current_slice['img_path_enhanced'])).get_fdata()[:,:,current_slice['slice']]
            # image_mean = np.mean(enhanced)
            # image_std = np.std(enhanced)
            # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            # enhanced = np.clip(enhanced, img_range[0], img_range[1])
            # if img_range[1] - img_range[0] == 0:
            #     divider = 1
            # else:
            #     divider = img_range[1] - img_range[0]
            enhanced = normalize_image(enhanced, -1, 1)
            return {"image": self.transform(img).float(), 
                    "label": torch.Tensor(np.array([label])), 
                    "enhanced": self.transform(enhanced).float(), 
                    "scar" : current_slice['scar'],
                    "img_path" : current_slice['img_path_original'],
                    "slice" : current_slice['slice']}
        
        label[label != 0] = 1 # unite the label for ROI
        return {"image": torch.Tensor(np.array([img])).float(), 
                "label": torch.Tensor(np.array([label])).float(), 
                "scar" : current_slice['scar'],
                "img_path" : current_slice['img_path_original'],
                "slice" : current_slice['slice']}

class MRMaskDataset(Dataset):
    def __init__(self, csv_path, mask_dir):
        self.csv_path = csv_path
        self.csv_file = pd.read_csv(self.csv_path)
        self.data = self.csv_file.to_dict(orient='records')
        self.data_dir = "/data1/marta"
        self.mask_dir = mask_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        current_slice = self.data[index]
        file_name = current_slice['img_path_original'].split("/")[-1].split('.')[0] + "_" + str(current_slice['slice']) + ".npy"
        mask = np.load(os.path.join(self.mask_dir, file_name))
        img = nib.load(os.path.join(self.data_dir, current_slice['img_path_original'])).get_fdata()[:,:,current_slice['slice']]
        # normalize original image
        # image_mean = np.mean(img)
        # image_std = np.std(img)
        # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        # img = normalize_image(img, img_range[0], img_range[1])
        # img = img / (img_range[1] - img_range[0])
        img = normalize_image(img, 0, 1)
        label = nib.load(os.path.join(self.data_dir, current_slice['label_path'])).get_fdata()[:,:,current_slice['slice']]
        enhanced = nib.load(os.path.join(self.data_dir, current_slice['img_path_enhanced'])).get_fdata()[:,:,current_slice['slice']]
        # normalize enhanced (target) image
        # image_mean = np.mean(enhanced)
        # image_std = np.std(enhanced)
        # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        # enhanced = normalize_image(enhanced, img_range[0], img_range[1])
        # if img_range[1] - img_range[0] == 0:
        #     divider = 1
        # else:
        #     divider = img_range[1] - img_range[0]
        # enhanced = enhanced / divider
        enhanced = normalize_image(enhanced, 0, 1)
        return {"image": self.transform(img).float(), # original image
                "label": torch.Tensor(np.array([label])), # actual annotated masks
                "enhanced": self.transform(enhanced).float(), # gamma corrected image (target)
                "scar" : current_slice['scar'],
                "slice" : current_slice['slice'],
                "roi_mask" : torch.Tensor(np.array(mask[0])), # binary mask for diffusion
                "path_original" : os.path.join(self.data_dir, current_slice['img_path_original'])
                }
    
class MRMaskDatasetUnet(Dataset):
    def __init__(self, csv_path, mask_dir):
        self.csv_path = csv_path
        self.csv_file = pd.read_csv(self.csv_path)
        self.data = self.csv_file.to_dict(orient='records')
        self.data_dir = "/data1/marta"
        self.mask_dir = mask_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        current_slice = self.data[index]
        file_name = current_slice['img_path_original'].split("/")[-1].split('.')[0] + "_" + str(current_slice['slice']) + ".npy"
        mask = np.load(os.path.join(self.mask_dir, file_name))
        img = nib.load(os.path.join(self.data_dir, current_slice['img_path_original'])).get_fdata()[:,:,current_slice['slice']]
        # normalize original image
        # image_mean = np.mean(img)
        # image_std = np.std(img)
        # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        # img = normalize_image(img, img_range[0], img_range[1])
        # img = img / (img_range[1] - img_range[0])
        img = normalize_image(img, 0, 1)
        label = nib.load(os.path.join(self.data_dir, current_slice['label_path'])).get_fdata()[:,:,current_slice['slice']]
        enhanced = nib.load(os.path.join(self.data_dir, current_slice['img_path_enhanced'])).get_fdata()[:,:,current_slice['slice']]
        # normalize enhanced (target) image
        # image_mean = np.mean(enhanced)
        # image_std = np.std(enhanced)
        # img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        # enhanced = normalize_image(enhanced, img_range[0], img_range[1])
        # if img_range[1] - img_range[0] == 0:
        #     divider = 1
        # else:
        #     divider = img_range[1] - img_range[0]
        # enhanced = enhanced / divider
        enhanced = normalize_image(enhanced, 0, 1)
        return {"image": self.transform(img).float(), 
                "label": torch.Tensor(np.array([label])), 
                "enhanced": self.transform(enhanced).float(), 
                "scar" : current_slice['scar'],
                "slice" : current_slice['slice'],
                "roi_mask" : torch.Tensor(np.array(mask[0])),
                "path_original" : os.path.join(self.data_dir, current_slice['img_path_original'])
                }
        