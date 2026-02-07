import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training DFA-Net')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config.device = device

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # dataset
    print(f"=> using dataset '{config.data.train_dataset}'")
    DATASET = datasets.__dict__[config.data.type](config)

    # DFA-Net config
    config.model.name = "DFA-Net"
    config.model.use_mdfa = args.use_mdfa
    config.model.use_lam = args.use_lam
    config.model.use_acfg = args.use_acfg
    config.model.dtcwt_levels = args.dtcwt_levels
    config.model.gamma_init = args.gamma_init

    # model
    print("=> creating DFA-Net diffusion model...")
    diffusion = DenoisingDiffusion(
        args=args,
        config=config,
        frequency_domain=True,
        cross_frequency_guidance=args.use_acfg
    )

    diffusion.train(DATASET)
