import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils.io import normalize_image

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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        current_slice = self.data[index]
        img = normalize_image(nib.load(os.path.join(self.data_dir, current_slice['img_path_original'])).get_fdata()[:,:,current_slice['slice']], 0, 1)
        label = nib.load(os.path.join(self.data_dir, current_slice['label_path'])).get_fdata()[:,:,current_slice['slice']]
        if self.mode == "enhancement":
            enhanced = nib.load(current_slice['img_path_enhanced']).get_fdata()[:,:,current_slice['slice']]
            return {"image": img, "label": label, "enhanced": enhanced, "scar" : current_slice['scar']}
        label_original = torch.Tensor(np.array([label]))
        label[label != 0] = 1 # unite the label for ROI
        return {"image": torch.Tensor(np.array([img])), 
                "label": torch.Tensor(np.array([label])), 
                "label_original" : label_original,
                "scar" : current_slice['scar'],
                "img_path" : current_slice['img_path_original'],
                "slice" : current_slice['slice']}