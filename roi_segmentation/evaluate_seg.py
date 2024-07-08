from dataset import MRDataset
from torch.utils.data import DataLoader
import torch
from UNet import SigmoidUNet
from utils.metrics import dice_score, mask_difference, recall_seg, iou_score
from tqdm import tqdm
import click
from utils.io import load_json
from typing import Dict, Final
import pandas as pd
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
    help='Path the where you want to output the csv.', 
    required=True,
)
@click.option(
    '--data_path', 
    '-d', 
    help='Path to the csv file with data you want to use for evaluation.', 
    required=False,
    default="../splits/val_slice.csv"
)
@click.option(
    '--postprocess', 
    '-p', 
    help='Whether or not you want to postprocess the mask.', 
    required=False,
    default=True
)

def main(model_path: str, data_path: str, output_path: str, postprocess: bool):
    cfg: Final[Dict] = load_json('config/cfg.json')
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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
        pred = unet(data['image'].float().to(device))
        pred = (pred > 0.5).float()
        if postprocess:
            pred = pred[0][0].cpu().numpy()
            pred = remove_small_areas(pred)
            pred = fill_gaps(pred)
            # pred = round_mask(pred)
            pred = fill_gaps_with_morphology(pred)
            pred = torch.Tensor(np.array([[pred]]))
        dice = dice_score(data['label'].to(device), pred.to(device)).item()
        diff = mask_difference(data['label'].to(device), pred.to(device)).item()
        recall = recall_seg(data['label'].to(device), pred.to(device)).item()
        iou = iou_score(data['label'].to(device), pred.to(device)).item()
        evaluation.append({
            "file_name" : data['img_path'][0],
            "dice" : dice,
            "diff" : diff,
            "slice" : data['slice'].item(),
            "recall" : recall,
            "iou":iou
        })
    df = pd.DataFrame(evaluation)
    df.to_csv(output_path)
    print("DICE: ", df['dice'].mean(), "std", df['dice'].std())
    print("RECALL: ", df['recall'].mean(), "std", df['recall'].std())
    print("IOU: ", df['iou'].mean(), "std", df['iou'].std())


if __name__ == '__main__':
    main()