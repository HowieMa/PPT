# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import pickle
import numpy as np

from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
from multiviews.cameras import unfold_camera_param


def build_multi_camera_system(cameras):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        cameras: list of camera parameters
    Returns:
        cams_system: a multi-cameras system
    """
    pymvg_cameras = []
    for (name, camera) in cameras:
        R, T, f, c, k, p = unfold_camera_param(camera)
        # intrinsic camera matrix K
        camera_matrix = np.array(
            [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)

        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = camera_matrix
        distortion = np.array([k[0], k[1], p[0], p[1], k[2]])
        distortion.shape = (5,)
        T = -np.matmul(R, T)
        M = camera_matrix.dot(np.concatenate((R, T), axis=1))       # KRT
        camera = CameraModel.load_camera_from_M(
            M, name=name, distortion_coefficients=distortion)
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-views 2d points

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    points_3d = camera_system.find3d(points_2d_set)
    return points_3d


def triangulate_poses(camera_params, poses2d, confs=None):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_params: a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: ndarray of shape nxkx2, len(cameras) == n      (#view, #joints, 2)
        confs:  (#view, #joints)
    Returns:
        poses3d: ndarray of shape n/nviews x k x 3
    """
    nviews = poses2d.shape[0]
    njoints = poses2d.shape[1]
    ninstances = len(camera_params) // nviews       # 1


    poses3d = []
    for i in range(ninstances):     # i = 0
        # Load 4 cameras
        cameras = []
        for j in range(nviews):
            camera_name = 'camera_{}'.format(j)
            cameras.append((camera_name, camera_params[i * nviews + j]))
        camera_system = build_multi_camera_system(cameras)

        #
        pose3d = np.zeros((njoints, 3))
        for k in range(njoints):        # for each joint (17 in total)
            points_2d_set = []

            # for current joint, only select views with high confidence to avoid error in triangulation
            # Adapted from https://github.com/yihui-he/epipolar-transformers/blob/master/vision/triangulation.py#L426
            conf_threshold = 0.85
            while True:
                selected_idx = np.where(confs[:, k] > conf_threshold)[0]
                if conf_threshold < -1:
                    break
                if len(selected_idx) <= 1:
                    conf_threshold -= 0.05
                    # print('conf too high, decrease to', conf_threshold)
                else:
                    break
            
            for j in range(nviews):
            #for j in selected_idx:
                camera_name = 'camera_{}'.format(j)
                points_2d = poses2d[i * nviews + j, k, :]
                points_2d_set.append((camera_name, points_2d))

            # triangulation for current joint
            pose3d[k, :] = triangulate_one_point(camera_system, points_2d_set).T
        poses3d.append(pose3d)
    return np.array(poses3d)
