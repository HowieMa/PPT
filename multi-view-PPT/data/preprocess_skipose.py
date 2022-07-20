import h5py
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import pickle 
import os 
import sys

data_root = sys.argv[1]


all_roots = {}
center_csv = os.path.join(data_root, 'ski_centers.csv')
for line in open(center_csv):
    fs = line.strip().split(' ')
    assert(len(fs)) == 5
    seq, frame = fs[0], fs[1]
    all_roots[seq + '_' + frame] = [float(fs[2]),float(fs[3]), float(fs[4])]



def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def cam2world(x, R, T):
    xcam = R.T.dot(x.T) + T  # rotate and translate R^-1 P_c + T
    return xcam.T

def pixel2cam2d(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] 
    y = (pixel_coord[:, 1] - c[1]) / f[1] 
    cam_coord = np.concatenate((x[:, None], y[:, None], np.ones(x[:, None].shape)), 1)
    return cam_coord




def fetch_and_format(h5_label_file, index):
    
    new_data = {}
    
    seq   = int(h5_label_file['seq'][index])
    cam   = int(h5_label_file['cam'][index])
    frame = int(h5_label_file['frame'][index])
    subj  = int(h5_label_file['subj'][index])
    pose_3D = h5_label_file['3D'][index].reshape([-1,3])
    pose_2D = h5_label_file['2D'][index].reshape([-1,2]) # in range 0..1
    pose_2D_px = 256*pose_2D # in pixels, range 0..255

    # camera parameters (unused here)
    K = h5_label_file['cam_intrinsic'][index].reshape([-1,3])
    T = h5_label_file['cam_position'][index].reshape(3,1)
    R = h5_label_file['R_cam_2_world'][index].reshape([3,3]).T

    # 
    root = np.asarray(all_roots[str(seq) + '_' + str(frame)]).reshape(3, 1)  # world coordinate
    root_cam = world2cam(root, R, T.reshape(3,1))   # (3, 1)
    pose_3D_cam = pose_3D + root_cam.reshape(1,3)   # (17, 3)
    
    pose_3D_world = cam2world(pose_3D_cam, R, T)
    
    
    image_name = 'seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq,cam,frame)
    
    # 
    fx, fy = 256.0 * K[0:1, 0], 256.0 * K[1:2, 1]
    cx, cy = 256.0 * K[0:1, 2], 256.0 * K[1:2, 2]
    cam_params = {'R':R, 'T': T, 
                  'fx': fx, 'fy': fy, 
                  'cx': cx, 'cy':cy, 
                  'k':np.zeros((3,)), 'p':np.zeros((2, ))
                 }
    
    new_data['joints_2d'] = pose_2D_px
    new_data['joints_3d_camera'] = pose_3D_cam
    new_data['joints_3d'] = pose_3D_world
    new_data['image'] = image_name
    new_data['joints_vis'] = np.ones((17, 3))

    new_data['camera'] = cam_params
    
    new_data['center'] = (128, 128)
    new_data['scale'] = (256 / 200, 256 / 200)    
    new_data['box'] = np.asarray([0, 0, 256, 256])
    
    # name
    new_data['source'] = 'skipose'
    new_data['subject'] = subj
    new_data['camera_id'] = cam
    new_data['image_id'] = frame
    new_data['video_id'] = seq
    new_data['action'] = 0
    new_data['subaction'] = 0

    return new_data



def format_data(split):

    label_file_name = os.path.join(data_root, 'Ski-PosePTZ-CameraDataset-png', split, 'labels.h5')

    h5_label_file = h5py.File(label_file_name, 'r')
    length = len(h5_label_file['seq'])
    print(length)
    all_data = []
    for i in range(length):
        item = fetch_and_format(h5_label_file, i)
        all_data.append(item)


    # save images
    os.system('mv {}/Ski-PosePTZ-CameraDataset-png/{}/seq* skipose/images/'.format(data_root, split))

    # save label
    if split == 'test':
        split = 'validation'        # change name
    with open('skipose/annot/ski_' + split + '.pkl', 'wb') as handle:
        pickle.dump(all_data, handle)


os.makedirs('skipose/annot/', exist_ok=True)
os.makedirs('skipose/images/', exist_ok=True)



format_data('train')
format_data('test')





