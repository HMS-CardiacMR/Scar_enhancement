import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from dataset import MRDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.nn import BCELoss
import click
from typing import Dict, Final
from utils.io import load_json
import wandb
from utils.metrics import dice_score
from tqdm import tqdm
from UNet import SigmoidUNet
from monai.losses import DiceLoss

@click.command()
@click.option(
    '--train_csv', 
    '-t', 
    help='Path to the csv file with train data information.', 
    required=False,
    default="../splits/train_slice.csv"
)
@click.option(
    '--val_csv', 
    '-v', 
    help='Path to the csv file with train data information.', 
    required=False,
    default="../splits/val_slice.csv"
)
@click.option(
    '--resume', 
    '-r', 
    help='Set True if you want to resume the training from a saved checkpoint.', 
    required=False,
    default=False
)
@click.option(
    '--resume_path', 
    '-c', 
    help='Path to the checkpoint to resume the training from.', 
    required=False,
    default=""
)
@click.option(
    '--results_path', 
    '-p', 
    help='Path where you want to store the saved checkpoints.', 
    required=True,
    default=""
)

def main(train_csv, val_csv, resume, resume_path, results_path):
    os.makedirs(results_path, exist_ok=True)
    cfg: Final[Dict] = load_json('config/cfg.json')
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(MRDataset(train_csv), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(MRDataset(val_csv), batch_size=cfg['batch_size'], shuffle=False)
    epochs = cfg['seg_num_epochs']

    channels = (64, 128, 256, 512, 1024)
    strides = (2, 2, 2, 2)
    unet = SigmoidUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
    ).to(device)
    
    if resume:
        unet.load_state_dict(torch.load(resume_path))

    loss_bce = BCELoss().to(device)
    loss_dice = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(unet.parameters(), lr=cfg["seg_learning_rate"])

    wandb.init(
        project = "roi-unet-training",
        config = {
            "learning_rate" : cfg["seg_learning_rate"],
            "epochs" : epochs,
            "channels" : channels,
            "strides" : strides
        }
    )

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train_loss = 0
        train_dice = 0
        val_loss = 0
        val_dice = 0
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = unet(data['image'].float().to(device))
            loss = 0.8 * loss_bce(outputs, data['label'].to(device)) + 0.2 * loss_dice(outputs, data['label'].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            cur_dice = dice_score(outputs, data['label'].to(device)).item()
            train_dice += cur_dice
            wandb.log({"batch_loss": loss.item(), "batch_dice": cur_dice})
        epoch_t_loss = train_loss / len(train_loader)
        epoch_t_dice = train_dice / len(train_loader)
        wandb.log({"epoch_loss": epoch_t_loss, "epoch_dice": epoch_t_dice})
        with torch.no_grad():
            for val_data in val_loader:
                output = unet(val_data['image'].float().to(device))
                val_loss += (0.8 * loss_bce(output, val_data['label'].to(device)) + 0.2 * loss_dice(output, val_data['label'].to(device)))
                val_dice += dice_score(output, val_data['label'].to(device)).item()
        wandb.log({"val_loss": val_loss/len(val_loader), "val_dice": val_dice/len(val_loader)})
        torch.save(unet.state_dict(), os.path.join(results_path, "roi_unet" + str(epoch) +".pt"))

if __name__ == '__main__':
    main()
