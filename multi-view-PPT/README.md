# Multi-view PPT for multi-view Human Pose Estimation

## Getting started

### Installation

1. Clone this repo, and we'll call the directory that you cloned multiview-pose as ${POSE_ROOT}   
~~~
git clone https://github.com/HowieMa/TransFusion-Pose.git
~~~

2. Install dependencies. 
~~~
pip install -r requirements.txt
~~~

3. Download pre-trained PPT-S on 256x256 resolution ([Google Drive](https://drive.google.com/file/d/17Lpo8F3wiTTvxhXRIlzVx1_3_GSBKsRa/view?usp=sharing)). 
Note: 
This model is different from the PPT-S trained on COCO.   
Due to the positional encoding and path projection layer of TokenPose, it is difficult to transfer model trained on one resolution to the other. 
The pre-trained PPT-S from single-view PPT is trained on images with `256x192` resolution. However, the resolution of human 3.6M is `256x256`. Thus, we have to train a new PPT-S based on images with  `256x256` resolution. 
Meanwhile, the order and definition of keypoints between COCO and Human 3.6M are slightly different, we also need to adjust the keypoint token. 

Please put them under `${POSE_ROOT}/multi-view-PPT/models`, and make them look like this:
~~~
${POSE_ROOT}/single-view-PPT/models
    └── ppt_s_ratio_07_coco_256_256.pth
~~~


### Data preparation
#### Human 3.6M
For Human36M data, please follow [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare images and annotations.


#### Ski-Pose
For Ski-Pose, please follow the instruction from their [website](https://www.epfl.ch/labs/cvlab/data/ski-poseptz-dataset/) to obtain the dataset.    
Once you download the **Ski-PosePTZ-CameraDataset-png.zip** and **ski_centers.csv**, unzip them and put into the same folder, named as ${SKI_ROOT}.    
Run `python data/preprocess_skipose.py ${SKI_ROOT}` to format it.   


Your folder should look like this:
~~~
${POSE_ROOT}
|-- data
|-- |-- h36m
    |-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        |-- images
            |-- s_01_act_02_subact_01_ca_01 
            |-- s_01_act_02_subact_01_ca_02

|-- |-- preprocess_skipose.py
|-- |-- skipose  
    |-- |-- annot
        |   |-- ski_train.pkl
        |   |-- ski_validation.pkl
        |-- images
            |-- seq_103 
            |-- seq_103
~~~


### Training and Testing
#### Human 3.6M
~~~
# Training
python run/pose2d/train.py --cfg experiments-local/h36m/ppt_multi/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3

# Evaluation (2D)
python run/pose2d/valid.py --cfg experiments-local/h36m/ppt_multi/256_fusion_enc3_GPE.yaml --gpus 0,1,2,3  

# Evaluation (3D)
python run/pose3d/estimate_tri.py --cfg experiments-local/h36m/ppt_multi/256_fusion_enc3_GPE.yaml
~~~

Our trained model can be found at this [link](https://drive.google.com/drive/folders/1y7ANiDeiIIC2hzrVVndTgYMLDD6Os1HG)




