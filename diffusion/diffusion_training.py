import os
import collections
import copy
import sys
import time
from random import seed
import torch

import numpy as np
from torch import optim

from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from unet import UNet, update_ema_params
from dataset import MRMaskDataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from scipy.io import savemat

torch.cuda.empty_cache()

#ROOT_DIR = "/mnt/alp/Users/Marta/"
ROOT_DIR = "/mnt/alp/Users/Burak/code/_enhancement_LGE/marta/"

#dir="/mnt/alp/Users/Marta/AnoDDPM/wand"
#wandb.init(
#    project = "roi-diffusion",
#    dir="/mnt/alp/Users/Burak/code/_enhancement_LGE/marta/wand"
#)

def train(training_dataset_loader, testting_dataset_loader, args, resume):
    """

    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """

    in_channels = 1
    if args["dataset"].lower() == "cifar" or args["dataset"].lower() == "leather":
        in_channels = 3

    if args["channels"] != "":
        in_channels = args["channels"]

    model = UNet()

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
        args['img_size'], 
        betas, 
        loss_weight=args['loss_weight'], 
        loss_type=args['loss-type'], 
        img_channels=in_channels
        )

    if resume:
        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])

        ema = UNet()
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']

    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])

    del resume
    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
    iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(200)
    # iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(150)

    # dataset loop
    for epoch in tqdm_epoch:
        print("EPOCH: ", epoch)
        mean_loss = []

        for i, data in enumerate(tqdm(training_dataset_loader)):
            if args["dataset"] == "cifar":
                # cifar outputs [data,class]
                x = data[0].to(device)
            else:
                y = data["enhanced"]
                y = y.to(device)
                x = data["image"]
                x = x.to(device)
                mask = data['roi_mask'].to(device)
                #mask = torch.ones(mask.size(), dtype=mask.dtype).to(device)

                #print(mask.dtype)
                #print(mask.shape)
                #savemat('asd1.mat', {'mask': mask.cpu().detach().numpy()})
                #mask = torch.ones(mask.size(), dtype=mask.dtype).to(device)
                #savemat('asd2.mat', {'mask': mask.cpu().detach().numpy()})

            loss, estimates = diffusion.p_loss(model, x, y, mask, args)

            noisy, est = estimates[1], estimates[2]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())
            #wandb.log({"batch_loss" : loss.data})

            row_size = 10
            if i == 0:
                for val_data in testting_dataset_loader:
                    y = val_data["enhanced"]
                    y = y.to(device)
                    x = val_data["image"]
                    x = x.to(device)
                    mask = val_data['roi_mask'].to(device)
                    #mask = torch.ones(mask.size(), dtype=mask.dtype).to(device)
                    training_outputs(diffusion, x, mask, y,  ema, args)
                    break

        losses.append(np.mean(mean_loss))
        time_taken = time.time() - start_time
        remaining_epochs = args['EPOCHS'] - epoch
        time_per_epoch = time_taken / (epoch + 1 - start_epoch)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)
        if epoch%5==0:
            vlb_terms = diffusion.calc_total_vlb(x, mask, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            #wandb.log({
            #    "priot_vib" : vlb_terms['prior_vlb'].mean(dim=-1).cpu().item(),
            #    "vb" : torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item(),
            #    "x_0_mse" : torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item(),
            #    "mse" : torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item()
            #})
        if epoch%10==0 and epoch !=0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    :param final: bool for final vs checkpoint
    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param ema: ema instance
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    if final:
        torch.save(
                {
                    'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(diffusion, x, mask, target, ema, args):
    """
    Saves video & images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param noisy: x_t
    :param epoch:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param ema: exponential moving average unet for sampling
    :param save_imgs: bool for saving imgs
    :param save_vids: bool for saving diffusion videos
    :return:
    """
    out = diffusion.forward_backward(ema, x, mask, "half", args['sample_distance'] // 2)
    #wandb.log({"output_image" : [wandb.Image(out[-1][0][0]), 
    #                                wandb.Image(out[0][0][0]),
    #                                wandb.Image(target[0][0].detach().cpu().numpy()), 
    #                                wandb.Image(out[1][0][0]),
    #                                wandb.Image(out[len(out)//2][0][0])]})

import re
def extract_number(filename):
    # Use regular expression to extract the numeric part of the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """
    # make directories
    torch.set_num_threads(4)
    for i in ['./model/',]:
        try:
            os.makedirs(i, exist_ok=True)
        except OSError:
            pass

    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # resume from final or resume from most recent checkpoint -> ran from specific slurm script?
    resume = 0
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args
    with open(f'{ROOT_DIR}diffusion/test_args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [ROOT_DIR + f'model/diff-params-ARGS={args["arg_num"]}',
              ROOT_DIR + f'model/diff-params-ARGS={args["arg_num"]}/checkpoint',
              ROOT_DIR + f'diffusion-videos/ARGS={args["arg_num"]}',
              ROOT_DIR + f'diffusion-training-images/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)
    if args["channels"] != "":
        in_channels = args["channels"]
    torch.set_num_threads(4)
    train_csv = os.path.join(ROOT_DIR, "splits/train_slice.csv")
    val_csv = os.path.join(ROOT_DIR, "splits/val_slice.csv")
    test_args = os.path.join(ROOT_DIR, "diffusion/test_args/args42.json")
    # if dataset is cifar, load different training & test set

    training_dataset_loader = DataLoader(MRMaskDataset(train_csv, "../data/masks/masks"), batch_size=16, shuffle=True, drop_last=True)
    testing_dataset_loader = DataLoader(MRMaskDataset(val_csv, "../data/masks/masks"), batch_size=1, shuffle=True)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    if resume:
        if resume == 1:
            checkpoints = os.listdir(ROOT_DIR + f'model/diff-params-ARGS={args["arg_num"]}/checkpoint')
            
            checkpoints = sorted(checkpoints, key=extract_number, reverse=True)
            for i in checkpoints:
                try:
                    file_dir = ROOT_DIR + f"model/diff-params-ARGS={args['arg_num']}/checkpoint/{i}"
                    print(file_dir)
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue

        else:
            file_dir = f'./model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    # load, pass args
    train(training_dataset_loader, testing_dataset_loader, args, loaded_model)

    # remove checkpoints after final_param is saved (due to storage requirements)
    for file_remove in os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'):
        os.remove(os.path.join(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint', file_remove))
    os.removedirs(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')


if __name__ == '__main__':
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    seed(1)

    main()