# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
# great thanks for fmassa!

# Run this python file with a config file, such as:
# python tools/compute_flops.py experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc5_mh8.yaml

import torch
import time
import torchvision

import numpy as np
import tqdm
import argparse

import _init_paths
from config import cfg
from config import update_config
import models

import numpy as np
import random
seed = 22

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda:0')


batch_size = 32


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args



def warmup(model, inputs, N=20):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()

def compute_fps(model, inputs, N=100):
    print("FPS calculation ...")
    warmup(model, inputs)
    s = time.time()
    for i in tqdm.tqdm(range(N)):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    print("FPS ", 1 / t)
    return t


def compute_throughput(model, inputs, ntest=100):
    print("ThroughPut calculation ...")
    warmup(model, inputs)
    start = time.time()
    for i in tqdm.tqdm(range(ntest)):
        out = model(inputs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batch_size * ntest / elapse
    print("Through-Put ", speed, "images / s")
    return speed


# configs from .yaml
args = parse_args()
update_config(cfg, args)


# create model >>>>>>>>>>>>>>>> 
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=True
)
model.to(device)
model.eval()


print("model params:{:.3f}M (/1000^2)".format(sum([p.numel() for p in model.parameters()])/1000**2))
with torch.no_grad():
    inputs = torch.randn(batch_size, 3, 256, 192).to(device)
    compute_throughput(model, inputs)

    # Note: when calculate FPS, we set the batch size to 1
    inputs = torch.randn(1, 3, 256, 192).to(device)
    compute_fps(model, inputs)
