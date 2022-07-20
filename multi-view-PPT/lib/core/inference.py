# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F

import numbers

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))        # (bs, num_joints, HW)
    idx = np.argmax(heatmaps_reshaped, 2)       # (bs, num_joints)  index of peak   value in 0 to HW-1
    maxvals = np.amax(heatmaps_reshaped, 2)     # (bs, num_joints)  value of peak location

    maxvals = maxvals.reshape((batch_size, num_joints, 1))  # (bs, num_joints, 1)
    idx = idx.reshape((batch_size, num_joints, 1))          # (bs, num_joints, 1)    value in 0 to HW-1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)      # (bs, num_joints, 2)    value in 0 to HW-1

    preds[:, :, 0] = (preds[:, :, 0]) % width                   # (bs, num_joints, 2) X
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)         # (bs, num_joints, 2) Y

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))    # (bs, num_joints, 2)
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    # batch_heatmaps: (bs, num_joints=17, 64, 64)
    # center: (bs, 2)       center in original image
    coords, maxvals = get_max_preds(batch_heatmaps)
    # coords: (bs, num_joints, 2)   location (X, Y) of peak in heatmap (64 * 64)

    heatmap_height = batch_heatmaps.shape[2]        # 64
    heatmap_width = batch_heatmaps.shape[3]         # 64

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]       # (64, 64)
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back location in heatmap (64 * 64) to location in (1000, 1000)
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    # radius is sigma
    # heatmap shape: 42 x 128 x 84
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L - 1)

    # def normalize(x, L):
    #     return -1. + 2. * (x.data + 0.5) / L
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    # affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    # theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    Iradius = int(radius + 0.5)
    affine_parameter = torch.zeros((num_pts, 2, 3), dtype=heatmap.dtype, device=heatmap.device)
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2
    # extract the sub-region heatmap
    grid_size = torch.Size([num_pts, 1, Iradius * 2 + 1, Iradius * 2 + 1])
    grid = F.affine_grid(affine_parameter, grid_size)
    # sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, mode='bilinear', padding_mode='reflection').squeeze(1)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros').squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, 0)

    X = torch.arange(-radius, radius + 0.0001, radius * 1.0 / Iradius, dtype=heatmap.dtype, device=heatmap.device).view(
        1, 1, Iradius * 2 + 1)
    Y = torch.arange(-radius, radius + 0.0001, radius * 1.0 / Iradius, dtype=heatmap.dtype, device=heatmap.device).view(
        1, Iradius * 2 + 1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1) + np.finfo(float).eps
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = pix2coord(x, downsample)
    y = pix2coord(y, downsample)
    return torch.stack([x, y], 1), score


def pix2coord(x, downsample):
    """convert pixels indices to real coordinates for 3D 2D projection
    """
    return x * downsample + downsample / 2.0 - 0.5