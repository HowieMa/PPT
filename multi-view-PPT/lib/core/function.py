# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
import random 


logger = logging.getLogger(__name__)


def get_keep_ratio(final_ratio, ite, total):
    # Start from 1.0, gradually decrease to the given ratio
    ratio = 1.0 - (1 - final_ratio) * ite / total
    return ratio




NUM_VIEW = {'multiview_h36m': 4, 'multiview_skipose': 6}

# pick the neighboring camera, same as Epipolar Transformers
cam_rank = {
    'multiview_h36m':
        {
            0: 2,  # cam1 -> cam3
            1: 3,  # cam2 -> cam4
            2: 0,  # cam3 -> cam1
            3: 1   # cam4 -> cam2
        },
    'multiview_skipose':
        {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4
        }
}


cam_pair = {
    'multiview_h36m': [[0, 2], [1, 3]],
    'multiview_skipose':[[0, 1], [2, 3], [4, 5]],
}


def get_epipolar_field(points1, center1, points2, center2, power=10, eps=1e-10):
    # Points1 / Points2: (B, N, 3)
    # Center1 / Center2: (B, 1, 3)
    # power: a higher value will generate a sharpen map along the epipolar line
    # Return: ()
    num_p1 = points1.shape[1]  # N1 = H * W

    # norm vector of  space C1C2P1 (Eq 3 in paper)
    vec_c1_c2 = center2 - center1 + eps         # (B, 1, 3)
    vec_c1_p1 = points1 - center1               # (B, N1, 3)
    space_norm_vec = torch.cross(vec_c1_p1, vec_c1_c2.repeat(1, num_p1, 1), dim=2) # (B, N, 3) x (B, N, 3) -> (B, N, 3)
    space_norm_vec_norm = F.normalize(space_norm_vec, dim=2, p=2)  # (B, N1, 3)

    vec_c2_p2 = points2 - center2  # (B, N2, 3)
    vec_c2_p2_norm = F.normalize(vec_c2_p2, dim=2, p=2)  # (B, N2, 3)

    # Eq 4 in paper
    cos = torch.bmm(space_norm_vec_norm, vec_c2_p2_norm.transpose(2, 1))    # (B, N1, 3) * (B, 3, N2) -> (B, N1, N2)

    field = 1 - cos.abs()
    field = field ** power
    field[field < 1e-5] = 1e-5      # avoid 0
    return field


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict):

    # total Epoch 
    total_epoch = config.TRAIN.END_EPOCH
    total_iter = len(data) * total_epoch
    cur_iter = epoch * len(data)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, weight, meta) in enumerate(data):
        # one subject, one action, in different views
        # input:    list, length:4, (bs, 3, 256, 256)       4 views
        # target:   list, length:4, (bs, 17, 64, 64)        4 views
        # weight:   list, length:4, (bs, 17, 1)             4 views
        data_time.update(time.time() - end)

        
        # ============= sample two views =============
        centers = [meta[idx]['cam_center'].float() for idx in range(len(input))]
        rays = [meta[idx]['rays'].float() for idx in range(len(input))]

        # ================== model forward ==================
        ratio = get_keep_ratio(0.7, cur_iter, total=total_iter)
        ratio = 0.7 
        output = model(input, centers=centers, rays=rays, ratio=ratio, fuse=True)  # list, (B, num_joints, H=64, W=64)

        # ================== Loss on the final heatmap (64 * 64) ==================
        loss = 0
        target_cuda = []
        for t, w, o in zip(target, weight, output):
            t = t.cuda(non_blocking=True)
            w = w.cuda(non_blocking=True)
            target_cuda.append(t)
            loss += criterion(o, t, w)
        target = target_cuda


        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.update(loss.item(), len(input) * input[0].size(0))

        # ================== accuracy based on heatmap (64 * 64) ==================
        nviews = len(output)
        acc = [None] * nviews
        cnt = [None] * nviews
        pre = [None] * nviews
        for j in range(nviews):
            _, acc[j], cnt[j], pre[j] = accuracy(
                output[j].detach().cpu().numpy(),
                target[j].detach().cpu().numpy())
        acc = np.mean(acc)
        cnt = np.mean(cnt)
        avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\tRatio {ratio:.3f}' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(data), ratio, batch_time=batch_time,
                      speed=len(input) * input[0].size(0) / batch_time.val,
                      data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage, ratio=ratio)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            for k in range(len(input)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'train'), view_name, i)
                save_debug_images(config, input[k], meta[k], target[k],
                                  pre[k] * 4, output[k], prefix)

    # epoch loss summary
    msg = 'Summary Epoch: [{0}]\tLoss ({loss.avg:.5f})\tAccuracy {acc.avg:.3f}'.format(epoch, loss=losses, acc=avg_acc)
    logger.info(msg)


def validate(config,
             loader,
             dataset,
             model,
             criterion,
             output_dir,
             writer_dict=None):

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    n_view = 6 if config.DATASET.TEST_DATASET == 'multiview_skipose' else 4
    nsamples = len(dataset) * n_view

    njoints = config.NETWORK.NUM_JOINTS                 # 17
    height = int(config.NETWORK.HEATMAP_SIZE[0])        # 64
    width = int(config.NETWORK.HEATMAP_SIZE[1])         # 64
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)      # (#sample, 17, 3)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)           # (#sample, 17,64, 64)

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, weight, meta) in enumerate(loader):
            # input:    list, length:4, (bs, 3, 256, 256)       4 views
            # target:   list, length:4, (bs, 17, 64, 64)        4 views
            # weight:   list, length:4, (bs, 17, 1)             4 views

            # ======================== combinations of input ========================
            batch_size = input[0].shape[0]

            rays = [meta[j]['rays'].float()  for j in range(len(input))  ] 
            centers = [meta[j]['cam_center'].float() for j in range(len(input))]
            output = model(input, centers=centers, rays=rays, ratio=0.7, fuse=True)

            
            # ======================== Loss calculation ========================
            loss = 0
            target_cuda = []
            for t, w, o in zip(target, weight, output):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                target_cuda.append(t)
                loss += criterion(o, t, w)

            target = target_cuda

            nimgs = len(output) * output[0].size(0)     # 4 cameras * batch_size
            losses.update(loss.item(), nimgs)

            # ================== accuracy based on heatmap (64 * 64) ==================
            nviews = len(output)
            acc = [None] * nviews
            cnt = [None] * nviews
            pre = [None] * nviews
            for j in range(nviews):
                _, acc[j], cnt[j], pre[j] = accuracy(
                    output[j].detach().cpu().numpy(),
                    target[j].detach().cpu().numpy())       # threshold: 64 / 10 * 0.5
            acc = np.mean(acc)
            cnt = np.mean(cnt)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            # ======================== Save prediction (heatmap + coords.) ========================
            preds = np.zeros((nimgs, njoints, 3), dtype=np.float32)     # (bs * #view, 17, 3)
            heatmaps = np.zeros(
                (nimgs, njoints, height, width), dtype=np.float32)      # (bs * #view, 17, 64, 64)
            for k, o, m in zip(range(nviews), output, meta):
                # o: (bs, 17, 64, 64)
                pred, maxval = get_final_preds(config,
                                               o.clone().cpu().numpy(),
                                               m['center'].numpy(),
                                               m['scale'].numpy())
                # pred:   (bs, num_joints=17, 2)    coordinate in original image (1000, 1000)
                # maxval: (bs, num_joints=17, 1)    peak value on heatmap
                pred = pred[:, :, 0:2]          # (bs, 17, 2)
                pred = np.concatenate((pred, maxval), axis=2)       # (bs, 17, 3)
                preds[k::nviews] = pred
                heatmaps[k::nviews] = o.clone().cpu().numpy()       # (bs, 17, 64, 64)

            all_preds[idx:idx + nimgs] = preds                      # (bs * #view, 17, 3) in original image
            all_heatmaps[idx:idx + nimgs] = heatmaps
            idx += nimgs

            # # ======================== Log ========================
            #if i % config.PRINT_FREQ == 0:
            if True:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time,
                          loss=losses, acc=avg_acc)
                logger.info(msg)

                for k in range(len(output)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    save_debug_images(config, input[k], meta[k], target[k],
                                      pre[k] * 4, output[k], prefix)

        #
        msg = '----Test (heatmap level)----: [{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    loss=losses, acc=avg_acc)
        logger.info(msg)

        # ======================= save all heatmaps and joint locations =======================
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['joint_names_order'] = a
        file.close()

        # ======================== evaluate JDF on 2D (based on 256 * 256)  ========================
        logger.info('Start 256 x 256 eval ....')
        name_value, perf_indicator = dataset.evaluate(all_preds)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')
        logger.info('Evaluate on 256 x 256 {}'.format(str(perf_indicator)))

    return perf_indicator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

