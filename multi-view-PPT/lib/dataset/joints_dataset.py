# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, affine_transform_pts


downsample = 16


class JointsDataset(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset

        self.root = cfg.DATASET.ROOT
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        self.transform = transform
        self.db = []

        self.num_joints = 17
        union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax',
            9: 'neck',
            10: 'upper neck',
            11: 'nose',
            12: 'head',
            13: 'head top',
            14: 'lsho',
            15: 'lelb',
            16: 'lwri',
            17: 'rsho',
            18: 'relb',
            19: 'rwri'
        }

        self.union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }


        self.actual_joints = {}
        self.u2a_mapping = {}

        # grid coordinate. For

        _y, _x = torch.meshgrid(torch.arange(self.image_size[0] // downsample),
                                torch.arange(self.image_size[1] // downsample))
        grid = torch.stack([_x, _y], dim=-1)  # Tensor, size:(32, 32, 2) val: 0-32
        grid = grid * downsample + downsample / 2.0 - 0.5  # Tensor, size:(32, 32, 2), val: 0-256
        self.grid = grid.view(-1, 2)  # Tensor, size:(hw, 2), val: 0-256

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        # ==================================== Image ====================================
        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                              db_rec['image'])
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        data_numpy = data_numpy[:1000]                  # According to ET

        # ==================================== Label ====================================
        joints = db_rec['joints_2d'].copy()             # (17, 2)   in original image scale (1000, 1000)
        joints_vis = db_rec['joints_vis'].copy()        # (17, 3)   0,0,0 or 1,1,1

        center = np.array(db_rec['center']).copy()      # (2, )     (cx, cy)  in original image scale
        scale = np.array(db_rec['scale']).copy()        # (2, )     (s1, s2)
        rotation = 0

        # ==================================== Camera  ====================================
        camera = db_rec['camera']
        # camera matrix
        R = camera['R'].copy()
        K = np.array([
            [float(camera['fx']), 0, float(camera['cx'])],
            [0, float(camera['fy']), float(camera['cy'])],
            [0, 0, 1.],
        ])
        T = camera['T'].copy()
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()
        cam_center = torch.Tensor(camera['T'].T)    # Tensor, (1, 3) camera center in world coordinate

        # fix the system error of camera
        # distCoeffs = np.array(
        #     [float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        # data_numpy = cv2.undistort(data_numpy, K, distCoeffs)
        # joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        # center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()

        # ==================================== Preprocess ====================================
        # augmentation factor
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

        # affine transformation matrix
        trans = get_affine_transform(center, scale, rotation, self.image_size)              # (2, 3)
        trans_inv = get_affine_transform(center, scale, rotation, self.image_size, inv=1)   # (2, 3)

        cropK = np.concatenate((trans, np.array([[0., 0., 1.]])), 0).dot(K)     # augmented K (for 256 * 256)
        KRT = cropK.dot(Rt)                 # (3,4)    camera matrix (intrinsic & extrinsic)

        # preprocess image  (1000, 1000) - > (256, 256)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)             # numpy image

        if self.transform:
            input = self.transform(input)

        # preprocess label  (1000, 1000) - > (256, 256)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)        # (17, 2) in (256, 256) scale
                if (np.min(joints[i, :2]) < 0 or
                        joints[i, 0] >= self.image_size[0] or
                        joints[i, 1] >= self.image_size[1]):
                    joints_vis[i, :] = 0

        # ========================== heatmap ==========================
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)                   # (17, 64, 64) heatmap
        target_weight = torch.from_numpy(target_weight)

        # ========================== 3D ray vectors ====================================
        # (256/down * 256/down, 3)
        coords_ray = self.create_3d_ray_coords(camera, trans_inv)

        # ==========================  Meta Info ==========================
        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],   # (17, 2) in origin image (1000, 1000)
            'joints_2d_transformed': joints,    # (17, 2) in input image (256, 256)
            'joints_vis': joints_vis,
            'source': db_rec['source'],

            'cam_center': cam_center,   # (1, 3) in world coordinate
            'rays': coords_ray,         # (256/down * 256/down, 3)  in world coordinate
            'KRT': KRT,                 # (3, 4) for augmented image
            'K': K,
            'RT': Rt,

            'img-path': image_file,                      # str
            'subject_id': db_rec['subject'],             # string 11
            'cam_id': db_rec['camera_id']                # string 01
        }
        return input, target, target_weight, meta

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1     # 13
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def create_3d_ray_coords(self, camera, trans_inv):
        multiplier = 1.0                        # avoid numerical instability
        grid = self.grid.clone()                # Tensor,   (hw, 2), val in 0-256
        # transform to original image R.T.dot(x.T) + T
        coords = affine_transform_pts(grid.numpy(), trans_inv)  # array, size: (hw, 2), val: 0-1000

        coords[:, 0] = (coords[:, 0] - camera['cx'][0]) / camera['fx'][0] * multiplier      # array
        coords[:, 1] = (coords[:, 1] - camera['cy'][0]) / camera['fy'][0] * multiplier

        # (hw, 3) 3D points in cam coord
        coords_cam = np.concatenate((coords,
                                     multiplier * np.ones((coords.shape[0], 1))), axis=1)   # array

        coords_world = (camera['R'].T @ coords_cam.T + camera['T']).T  # (hw, 3)    in world coordinate    array
        coords_world = torch.from_numpy(coords_world).float()  # (hw, 3)
        return coords_world
