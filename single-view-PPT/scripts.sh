

### Training ###
## COCO
# PPT-S
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml

# PPT-B
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --cfg experiments/coco/ppt/ppt_b_256_192_patch43_dim192_depth12_heads8.yaml

# PPT-L/D6
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --cfg experiments/coco/ppt/ppt_L_D6_256_192_patch43_dim192_depth12_heads8.yaml





### Test ###

# PPT-S
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml TEST.MODEL_FILE models/pytorch/pose_coco/ppt_s_256x192_ratio07.pth TEST.USE_GT_BBOX False

# PPT-B
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/ppt/ppt_b_256_192_patch43_dim192_depth12_heads8.yaml TEST.MODEL_FILE models/pytorch/pose_coco/ppt_b_256x192_ratio07.pth TEST.USE_GT_BBOX False

# PPT-L/D6
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/ppt/ppt_L_D6_256_192_patch43_dim192_depth6_heads8.yaml TEST.MODEL_FILE models/pytorch/pose_coco/ppt_l_d6_256x192_ratio07.pth TEST.USE_GT_BBOX False


# We also add the evaluation scripts of other popular networks (HRNet, TransPose, TokenPose, etc), which can help you compare with other methods easily.
# HR-Net-W32
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth TEST.USE_GT_BBOX False

# HR-Net-W48
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth TEST.USE_GT_BBOX False

# TransPose-R-A3
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml TEST.MODEL_FILE models/pytorch/pose_coco/tp_h_48_256x192_enc6_d96_h192_mh1.pth TEST.USE_GT_BBOX False

# TransPose-H-A6
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml TEST.MODEL_FILE models/pytorch/pose_coco/tp_h_48_256x192_enc6_d96_h192_mh1.pth TEST.USE_GT_BBOX False

# TokenPose-S
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/coco/tokenpose/tokenpose_s_v1_256_192_patch43_dim192_depth12_heads8.yaml TEST.USE_GT_BBOX False


### ThroughPut calculation ###
# PPT-S
python tools/compute_fps.py --cfg experiments/coco/ppt/ppt_s_v1_256_192_patch43_dim192_depth12_heads8.yaml
# PPT-B
python tools/compute_fps.py --cfg experiments/coco/ppt/ppt_b_256_192_patch43_dim192_depth12_heads8.yaml
# PPT-L/D6
python tools/compute_fps.py --cfg experiments/coco/ppt/ppt_L_D6_256_192_patch43_dim192_depth6_heads8.yaml

## For a fair comparison of FPS / ThroughPut, we suggest you run the following scripts on the same machine. 
# TokenPose-S
python tools/compute_fps.py --cfg experiments/coco/tokenpose/tokenpose_s_v1_256_192_patch43_dim192_depth12_heads8.yaml
# TokenPose-B
python tools/compute_fps.py --cfg experiments/coco/tokenpose/tokenpose_b_256_192_patch43_dim192_depth12_heads8.yaml
# TokenPose-L/D6
python tools/compute_fps.py --cfg experiments/coco/tokenpose/tokenpose_L_D6_256_192_patch43_dim192_depth12_heads8.yaml

# HR-Net-W32
python tools/compute_fps.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
# HR-Net-W48
python tools/compute_fps.py --cfg experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml

# TransPose-R-A3
python tools/compute_fps.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml
# TransPose-H-A6
python tools/compute_fps.py --cfg experiments/coco/transpose_h/TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml