import os
from dataset import MRDataset
from torch.utils.data import DataLoader
import torch
from UNet import SigmoidUNet
from tqdm import tqdm
import click
from utils.io import load_json
from typing import Dict, Final
import numpy as np
from utils.io import fill_gaps_with_morphology, remove_small_areas, fill_gaps

@click.command()
@click.option(
    '--model_path', 
    '-m', 
    help='Path to the model you want to evaluate.', 
    required=True,
)
@click.option(
    '--output_path', 
    '-o', 
    help='Path the where you want to output the masks.', 
    required=True,
)
@click.option(
    '--data_path', 
    '-d', 
    help='Path to the csv file with data you want to use for evaluation.', 
    required=False,
    default="../splits/other.csv"
)
@click.option(
    '--postprocess', 
    '-p', 
    help='Whether or not you want to postprocess the mask.', 
    required=False,
    default="True"
)

def main(model_path: str, data_path: str, output_path: str, postprocess: bool):
    cfg: Final[Dict] = load_json('config/cfg.json')
    os.makedirs(output_path, exist_ok=True)
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    data_loader = DataLoader(MRDataset(data_path), batch_size=1, shuffle=False)
    channels = (64, 128, 256, 512, 1024)
    strides = (2, 2, 2, 2)
    unet = SigmoidUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
    ).to(device)
    unet.load_state_dict(torch.load(model_path))
    evaluation = []
    for data in tqdm(data_loader):
        print(data['image'].shape)
        pred = unet(data['image'].float().to(device))
        pred = (pred > 0.5).float()
        if postprocess:
            pred = pred[0][0].cpu().numpy()
            pred = remove_small_areas(pred)
            pred = fill_gaps(pred)
            # pred = round_mask(pred)
            pred = fill_gaps_with_morphology(pred)
            pred = torch.Tensor(np.array([[pred]]))
        file_name = data['img_path'][0].split('/')[-1].split('.')[0] + "_" + str(data['slice'].item()) + ".npy"
        out_path = os.path.join(output_path, file_name)
        np.save(out_path, pred)


if __name__ == '__main__':
    main()