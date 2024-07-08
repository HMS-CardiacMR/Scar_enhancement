import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import json
import torch
from dataset import MRMaskDataset
import click
from unet import UNet, update_ema_params
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from torch.utils.data import DataLoader
from matplotlib import animation
import matplotlib.pyplot as plt
from helpers import gridify_output, load_parameters
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import nibabel as nib
from utils.data_utils import *

#ROOT_DIR = "/mnt/alp/Users/Marta/clean_code"
ROOT_DIR = "/mnt/alp/Users/Burak/code/_enhancement_LGE/marta/"

@click.command()
@click.option(
    '--model_path', 
    '-m', 
    help='Path to the csv file with train data information.', 
    required=True,
)
@click.option(
    '--output_path', 
    '-o', 
    help='Path to the csv file with train data information.', 
    required=True,
)
@click.option(
    '--repeat', 
    '-r', 
    help='Path to the csv file with train data information.', 
    required=False,
    default=5
)
@click.option(
    '--distance', 
    '-t', 
    help='t distance.', 
    required=False,
    default=250
)

def main(model_path, output_path, repeat, distance):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced_scar"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced_no_scar"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "source"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "target"), exist_ok=True)
    test_args = os.path.join(ROOT_DIR, "diffusion/test_args/args42.json")
    val_csv = os.path.join(ROOT_DIR, "splits/test_slice_small.csv")
    #val_loader = DataLoader(MRMaskDataset(val_csv, "../data/masks_other"), batch_size=1, shuffle=False)
    val_loader = DataLoader(MRMaskDataset(val_csv, "../data/masks/masks"), batch_size=1, shuffle=False)
    with open(test_args, 'r') as f:
        args = json.load(f)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    in_channels = 1

    unet = UNet()

    
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
        args['img_size'], 
        betas, 
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], 
        img_channels=in_channels)
    
    # load the model
    loaded_model = torch.load(model_path, map_location=device)
    unet.load_state_dict(loaded_model["ema"])
    unet.to(device)
    unet.eval()

    times = []
    for j, val_data in enumerate(tqdm(val_loader)):
        for r in range(repeat):
            start_time = time.time()

            ROI_mask = val_data['roi_mask'].to(device)
            #ROI_mask = torch.ones(ROI_mask.size(), dtype=ROI_mask.dtype).to(device)
            output = diffusion.forward_backward(
                unet, val_data['image'].to(device), ROI_mask, see_whole_sequence="half",
                t_distance=distance
            )
            end_time = time.time() - start_time
            times.append(end_time)
            # save it as nifti for image editing
            get_gt_vals = nib.load(val_data['path_original'][0]).get_fdata()[:,:,val_data['slice']]
            out_norm = normalize_image(output[-1][0][0].detach().cpu().numpy(), get_gt_vals.min(), get_gt_vals.max())
            enh = normalize_image(val_data['enhanced'][0][0].numpy(), get_gt_vals.min(), get_gt_vals.max())
            src = normalize_image(val_data['image'][0][0].detach().cpu().numpy(), get_gt_vals.min(), get_gt_vals.max())
            affine = nib.load(val_data['path_original'][0]).affine
            img = nib.Nifti1Image(out_norm, affine)
            img_name = val_data['path_original'][0].split("/")[-1][:16] + "_" + str(val_data['slice'].item())
            # for i, out in enumerate(output):
            if val_data['scar']:
                result_folder = os.path.join(output_path, "enhanced_scar")
                plt.axis("off")
                plt.imshow(out_norm, cmap="gray", vmax=0.9*get_gt_vals.max())
                plt.savefig(os.path.join(result_folder, img_name  + "_" + str(r) + ".png"), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                nib.save(img, os.path.join(result_folder, img_name.split(".nii")[0] + "_" + str(r) + ".nii.gz"))    
            else:
                result_folder = os.path.join(output_path, "enhanced_no_scar")
                plt.axis("off")
                plt.imshow(out_norm, cmap="gray")
                plt.savefig(os.path.join(result_folder, img_name + "_" + str(r) + ".png"), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()
                nib.save(img, os.path.join(result_folder, img_name + "_" + str(r) + ".nii.gz"))    
        target_path = os.path.join(output_path, "target")
        plt.axis("off")
        plt.imshow(enh, cmap="gray", vmax=0.9*get_gt_vals.max())
        plt.savefig(os.path.join(target_path, img_name + ".png"), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        nib.save(nib.Nifti1Image(enh, affine), os.path.join(target_path, img_name + "_" + str(r) + ".nii.gz"))    
        source_path = os.path.join(output_path, "source")
        plt.axis("off")
        plt.imshow(src, cmap="gray", vmax=0.9*get_gt_vals.max())
        plt.savefig(os.path.join(source_path, img_name + ".png"), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        nib.save(nib.Nifti1Image(src, affine), os.path.join(source_path, img_name + "_" + str(r) + ".nii.gz"))    
        if j == 500:
            break
    
    times_dict = {"time" : times}
    df = pd.DataFrame(times_dict)
    df.to_csv("times_testt_" + str(distance) + ".csv")

       
        # x = diffusion.sample_q(val_data['image'], torch.tensor([1]), torch.randn_like(val_data['image']))
        # plt.axis('off')
        # plt.imshow(x[0][0], cmap='gray')
        # plt.savefig("noised.png")

        # x = val_data['image']

        # for t in tqdm(range(1000)):
        #     t_batch = torch.tensor([t]).repeat(x.shape[0])
        #     noise = torch.randn_like(x)
        #     # noise = self.noise_fn(x, t_batch).float()
        #     x = diffusion.sample_q_gradual(x, t_batch, noise)
        #     plt.axis('off')
        #     plt.imshow(x[0][0], cmap='gray')
        #     plt.savefig("noised" + str(t) + ".png")


            # for i, out in enumerate(tqdm(output)):
            #     plt.axis("off")
            #     plt.imshow(out[0][0], cmap='gray')
            #     plt.savefig(str(i) + ".png")
            #     plt.close()
        # break



    # for i, data in enumerate(tqdm(val_loader)):
    #     original = data['image'].to(device)
    #     enhanced = data['enhanced'].to(device)
    #     loss, pred = diffusion.p_loss(unet, original, enhanced, args)
    #     pred = pred[2][0][0].detach().cpu().numpy()
    #     original = original[0][0].detach().cpu().numpy()
    #     enhanced = enhanced[0][0].detach().cpu().numpy()
        # save predicted image
        # plt.axis('off')
        # plt.imshow(pred, cmap="gray")
        # if data['scar']:
        #     plt.savefig(os.path.join(output_path, "enhanced_scar/" + str(i) + ".png"), 
        #                 bbox_inches='tight', pad_inches=0, transparent=True)
        # else:
        #     plt.savefig(os.path.join(output_path, "enhanced_no_scar/" + str(i) + ".png"), 
        #                 bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.close()
        # # save target image
        # plt.axis('off')
        # plt.imshow(enhanced, cmap="gray")
        # plt.savefig(os.path.join(output_path, "target/" + str(i) + ".png"), 
        #                 bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.close()
        # # save source image
        # plt.axis('off')
        # plt.imshow(original, cmap="gray")
        # plt.savefig(os.path.join(output_path, "source/" + str(i) + ".png"), 
        #                 bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.close()

if __name__ == '__main__':
    main()