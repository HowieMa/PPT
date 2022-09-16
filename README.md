# token-Pruned Pose Transformer 
"PPT: token-Pruned Pose Transformer for monocular and multi-view human pose estimation"
Haoyu Ma, Zhe Wang, Yifei Chen, Deying Kong, Liangjian Chen, Xingwei Liu, Xiangyi Yan, Hao Tang, and Xiaohui Xie.   
In ECCV 2022


## Introduction

* We propose the token-Pruned Pose Transformer (PPT) for efficient 2D human pose estimation, which can locate the human body area and prune background tokens with the help of a Human Token Identification module. 
    
* We propose the strategy of "Human area fusion" for multi-view pose estimation. Built upon PPT, the multi-view PPT can efficiently fuse cues from human areas of multiple views. 
 

![framework](https://github.com/HowieMa/PPT/blob/main/images/framework.png)



## Running
For monocular 2D pose estimation, please see [single-view-PPT](https://github.com/HowieMa/PPT/tree/main/single-view-PPT).     
For multi-view 3D pose estimation, please see [multi-view-PPT](https://github.com/HowieMa/PPT/tree/main/multi-view-PPT).   



## Citation
If you find our code helps your research, please cite the paper:

~~~
@inproceedings{ma2022ppt,
  title={PPT: token-Pruned Pose Transformer for monocular and multi-view human pose estimation},
  author={Ma, Haoyu and Wang, Zhe and Chen, Yifei and Kong, Deying and Chen, Liangjian and Liu, Xingwei and Yan, Xiangyi and Tang, Hao and Xie, Xiaohui},
  booktitle={ECCV},
  year={2022}
}
~~~



## Acknowledgement
* [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [TransPose](https://github.com/yangsenius/TransPose)
* [TokenPose](https://github.com/leeyegy/TokenPose)

* [Cross-view Fusion](https://github.com/microsoft/multiview-human-pose-estimation-pytorch)
* [Epipolar Transformer](https://github.com/yihui-he/epipolar-transformers)  
* [TransFusion](https://github.com/HowieMa/TransFusion-Pose)

