# PPT for single-view 2D Human Pose Estimation

## Getting started

### Installation

1. Clone this repo, and we'll call the directory that you cloned multiview-pose as ${POSE_ROOT}, and enter the path of single-view 2D pose estimation   
~~~
git clone https://github.com/HowieMa/PPT.git
cd single-view-PPT/
~~~

2. Install dependencies. 
~~~
pip install -r requirements.txt
~~~

3. Download pretrained models from our model zoo: [Google Drive](https://drive.google.com/drive/folders/1GEzXEmwZKX7g6u55n7r-x3e7lCvrLzV6?usp=sharing). 
we also include the pre-trained HRNet, TransPose, and TokenPose (rerun by us) into the same folder to simplify the downloading. 
**All copyright belongs to their respective owners!**. You can also download them from their original source separately:    
[HRNet model zoo](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC)   
[TransPose model zoo](https://github.com/yangsenius/TransPose)   

Please put them under `${POSE_ROOT}/single-view-PPT/models`, and make them look like this:
~~~
${POSE_ROOT}/single-view-PPT/models
└── pytorch
    └── imagenet
        └── hrnet_w32-36af842e.pth
        └── hrnet_w48-8ef0771d.pth
        └── ...
    └── pose_coco
        └── ppt_s_256x192_ratio07.pth
        └── ppt_b_256x192_ratio07.pth
        └── ppt_l_d6_256x192_ratio07.pth
        └── ... 
~~~


4. Data Preparation 
We follow the steps of HRNet to prepare the COCO train/val/test dataset and the annotations. The detected person results are downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk). Please download or link them to `${POSE_ROOT}/data/coco/`, and make them look like this
~~~
${POSE_ROOT}/single-view-PPT/data/coco/
|-- annotations
|   |-- person_keypoints_train2017.json
|   `-- person_keypoints_val2017.json
|-- person_detection_results
|   |-- COCO_val2017_detections_AP_H_56_person.json
|   `-- COCO_test-dev2017_detections_AP_H_609_person.json
`-- images
	|-- train2017
	|   |-- 000000000009.jpg
	|   |-- ... 
	`-- val2017
		|-- 000000000139.jpg
		|-- ... 
~~~

### Training & Testing

1. Testing on COCO val2017 dataset
~~~
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml TEST.MODEL_FILE models/pytorch/pose_coco/ppt_s_256x192_ratio07.pth TEST.USE_GT_BBOX False
~~~
This script should create folders `{POSE_ROOT}/single-view-PPT/output/coco/pose_tokenpose_s/ppt_s_v1_256_192_patch43_dim192_depth12_heads8/`   
The result should be 


| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |  
|---|---|---|---|---|---|---|---|---|---|---|  
| PPT-s | 0.722 | 0.890 | 0.797 | 0.686 | 0.793 | 0.778 | 0.930 | 0.845 | 0.733 | 0.842 |  



2. Compute FPS and Throughput 
~~~
python tools/compute_fps.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml
~~~ 
On a single 2080Ti, the FPS should be `130` and the throughput should be `870` images/s
Note that, the results may be slightly different


3. Training on COCO train2017 dataset
~~~
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml
~~~


For all scripts, please refer to [scripts.sh](https://github.com/HowieMa/PPT/tree/main/single-view-PPT/scripts.sh)