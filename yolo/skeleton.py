"""
File: Simple deeplearning test project
"""
"""
Import Section of Common Libs
"""
import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

"""
Import Section of Scientific Libs
"""
import math
import numpy as np
import pandas as pandas
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.autograd import variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.torch_utlis import select_device, torch_distributed_zero_first

"""
Constants Section
"""
ROOT = Path(os.path.relpath(Path(__file__).resolve(), Path.cwd()))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5l.pt', helpt='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size(pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in ram(default) or disk')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 ro 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epislon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience(epochs without improvement)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    opt = parse_opt(True)

    # Resume

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch_size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, 'evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
