

# ------------------------------------------------------------------------------
# Modified by Yanjie Li (leeyegy@gmail.com)
# TokenPose + Sparse for 2D single person PE
# Multi-view
# cross-view Fusion
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math

import os
import logging

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

logger = logging.getLogger(__name__)


# ******************************************************************************************
# ************************************** Token-Pose **************************************
# ******************************************************************************************


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask = None, return_tok=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)     # (B, N, C)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)       # 3 * (B, H, N, C/H)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale       #  (B, H, N, C/H) @ (B, H, C/H, N) -> (B, H, N, N)

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max                   # -INF
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)         # # (B, H, N, N)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)      # (B, H, N, N) * (B, H, N, C/H)
        out = rearrange(out, 'b h n d -> b n (h d)')        # (B, H, N, C/H) -> (B, N, C)
        out =  self.to_out(out)                             # (B, N, C)

        if return_tok:
            # N = HW + J
            J = self.num_keypoints
            tok_attn = attn[:, :, :J, J:]                        # (B, H, J, HW)
            tok_attn = tok_attn.sum(1) / self.heads              # (B, J, HW), average all head 
            return [out, tok_attn]
        else:
            return out



def batched_index_select(input, dim, index):
    # input:(B, C, HW). index(B, N)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)      # (B,C, HW)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False, pruning_loc=[3,6,9]):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head)),       # without residual
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))

            ]))

        # >>>>>>>>>>>>>>>>>>>>>
        self.pruning_loc = pruning_loc
        # <<<<<<<<<<<<<<<<<<<<<

    def forward(self, x, mask = None,pos=None, prune=False, keep_ratio=0.7):
        B = x.shape[0]
        pos = pos.expand(B, -1, -1)
        for idx,(attn, ff) in enumerate(self.layers):
            # >>>>>>>>>> add patch embedding >>>>>>>>>>
            if idx>0 and self.all_attn:
                x[:,self.num_keypoints:] += pos
            
            # >>>>>>>>>> Attention layer >>>>>>>>>>
            x_att, tok_attn = attn(x, mask=mask, return_tok=True)           # 
            # x_att: (B, HW+J, C)
            # tok_attn: (B J, HW)
            x = x_att + x                                                   # (B, J+Hw, C)

            # >>>>>>>>>>>>>>>>>>>>> real prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if idx in self.pruning_loc and prune: 
                joint_tok_copy = x[:, :self.num_keypoints]                           # (B, J, C)     save token

                B, _, num_patches = tok_attn.shape                          # num_patch = HW
                num_keep_node = math.ceil( num_patches * keep_ratio )       # K = HW r

                # attentive token
                human_attn = tok_attn.sum(1)           # (B, HW)
                attentive_idx = human_attn.topk(num_keep_node, dim=1)[1]            # (B, K)        without gradient
                x_attentive = batched_index_select(x[:, self.num_keypoints:], 1, attentive_idx)      # (B, N, C) -> (B, K, C)
                pos =  batched_index_select(pos, 1, attentive_idx)                               # (B, N, C) -> (B, K, C)
                
                x = torch.cat([joint_tok_copy, x_attentive], dim=1)                      # (B, J+K, C)

            # >>>>>>>>>> MLP layer >>>>>>>>>>
            x = ff(x)

            # x = attn(x, mask = mask)
            # x = ff(x)
        return x


class TransformerMulti(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False, pruning_loc=[3,6,9]):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head)),       # without residual
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))

            ]))

        # >>>>>>>>>>>>>>>>>>>>>
        self.pruning_loc = pruning_loc
        # <<<<<<<<<<<<<<<<<<<<<

    def forward_single(self, x, mask=None, pos=None, prune=False, keep_ratio=0.7):
        B = x.shape[0]
        pos = pos.expand(B, -1, -1)
        for idx,(attn, ff) in enumerate(self.layers):
            # >>>>>>>>>> add patch embedding >>>>>>>>>>
            if idx>0 and self.all_attn:
                x[:,self.num_keypoints:] += pos
            
            # >>>>>>>>>> Attention layer >>>>>>>>>>
            x_att, tok_attn = attn(x, mask=mask, return_tok=True)           # 
            # x_att: (B, HW+J, C)
            # tok_attn: (B J, HW)
            x = x_att + x                                                   # (B, J+Hw, C)

            # >>>>>>>>>>>>>>>>>>>>> real prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if idx in self.pruning_loc and prune: 
                joint_tok_copy = x[:, :self.num_keypoints]                           # (B, J, C)     save token

                B, _, num_patches = tok_attn.shape                          # num_patch = HW
                num_keep_node = math.ceil( num_patches * keep_ratio )       # K = HW r

                # attentive token
                human_attn = tok_attn.sum(1)           # (B, HW)
                attentive_idx = human_attn.topk(num_keep_node, dim=1)[1]            # (B, K)        without gradient
                x_attentive = batched_index_select(x[:, self.num_keypoints:], 1, attentive_idx)      # (B, N, C) -> (B, K, C)
                pos =  batched_index_select(pos, 1, attentive_idx)                               # (B, N, C) -> (B, K, C)
                
                x = torch.cat([joint_tok_copy, x_attentive], dim=1)                      # (B, J+K, C)

            # >>>>>>>>>> MLP layer >>>>>>>>>>
            x = ff(x)

            # x = attn(x, mask = mask)
            # x = ff(x)
        return x

    def forward(self, xs, mask = None, poss=None, prune=False, keep_ratio=0.7, pos_emb_3ds=None):
        # x: a list, size: (B, J+HW, C)
        # pos: a list, size: (B, HW, C)
        # pos_emb_3ds: a list, size: (B, HW, C)

        if not isinstance(xs, list):
            return self.forward_single(xs, mask, poss, prune, keep_ratio)

        features = []
        prune_pos = []
        prune_pos_3d = []
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1-10 Layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for x, pos, p3d in zip(xs, poss, pos_emb_3ds):

            for idx,(attn, ff) in enumerate(self.layers[:10]):
                # >>>>>>>>>> add patch embedding >>>>>>>>>>
                if idx>0 and self.all_attn:
                    x[:,self.num_keypoints:] += pos
                
                # >>>>>>>>>> Attention layer >>>>>>>>>>
                x_att, tok_attn = attn(x, mask=mask, return_tok=True)           # 
                # x_att: (B, HW+J, C)
                # tok_attn: (B J, HW)
                x = x_att + x                                                   # (B, J+Hw, C)

                # >>>>>>>>>>>>>>>>>>>>> real prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if idx in self.pruning_loc and prune:       # 3,6, 9 layers
                    joint_tok_copy = x[:, :self.num_keypoints]                           # (B, J, C)     save token

                    B, _, num_patches = tok_attn.shape                          # num_patch = HW
                    num_keep_node = math.ceil( num_patches * keep_ratio )       # K = HW r

                    # attentive token
                    human_attn = tok_attn.sum(1)           # (B, HW)
                    attentive_idx = human_attn.topk(num_keep_node, dim=1)[1]            # (B, K)        without gradient
                    x_attentive = batched_index_select(x[:, self.num_keypoints:], 1, attentive_idx)     # (B, N, C) -> (B, K, C)
                    pos =  batched_index_select(pos, 1, attentive_idx)                                  # (B, N, C) -> (B, K, C)
                    p3d = batched_index_select(p3d, 1, attentive_idx)                                   # (B, N, C) -> (B, K, C)
                    
                    x = torch.cat([joint_tok_copy, x_attentive], dim=1)                      # (B, J+K, C)

                # >>>>>>>>>> MLP layer >>>>>>>>>>
                x = ff(x)

                # x = attn(x, mask = mask)
                # x = ff(x)
            features.append(x)          # (B, J+K, C)
            prune_pos.append(pos)       # (B, K, C)
            prune_pos_3d.append(p3d)    # (B, K, C)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> split and group >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        joint_queries = []
        pruned_visual_features = []
        for x in features:
            joint_queries.append(x[:, :self.num_keypoints])         # (B, J, C)
            pruned_visual_features.append(x[:, self.num_keypoints: ] )     # (B, K, C)

        num_view = len(features)            # M views in total (paper notation)

        pruned_features = torch.cat(pruned_visual_features, dim=1)      # (B, K * M, C)
        joint_queries = torch.cat(joint_queries, dim=1)                 # (B, J * M, C)

        x = torch.cat([joint_queries, pruned_features], dim=1)          # (B, JV + KV, C)   

        prune_pos = torch.cat(prune_pos, dim=1)                         # (B, K * M, C)
        prune_pos_3d = torch.cat(prune_pos_3d, dim=1)                   # (B, K * M, C)

        # add this line to enable 3D PE in the last layers. Otherwise, only 2D PE 
        prune_pos = prune_pos + prune_pos_3d        

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 11-12 Layers (Cross-view Fusion) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for idx, (attn, ff) in enumerate(self.layers[10:]):
            if idx + 10 > 0 and self.all_attn:
                x[:, self.num_keypoints * num_view : ] += prune_pos
            # >>>>>>>>>> Attention layer >>>>>>>>>>
            x_att, _ = attn(x, mask=mask, return_tok=True)      # (B, JV+KV, C)
            x = x_att + x   # (B, JV+KV, C)
            # >>>>>>>>>> MLP layer >>>>>>>>>>
            x = ff(x)

        outputs = x[:, : self.num_keypoints * num_view]    # (B, JV)
        outputs = torch.chunk(outputs, num_view, dim=1)  # list, (B, V), (B, V)

        return outputs


# ******************************************************************************************
# ************************************** TokenPose-multi **************************************
# ******************************************************************************************



class TokenPose_S_base_multi(nn.Module):
    def __init__(self, *, image_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, hidden_heatmap_dim=64*6,heatmap_dim=64*48,heatmap_size=[64,48], channels = 3, dropout = 0., emb_dropout = 0.,pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(image_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size[0] // (4*patch_size[0])) * (image_size[1] // (4*patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pos_embedding_type in ['sine','none','learnable','sine-full']

        # >>>>>>>>>>>>>>>>>>>>>>>>> View Token >>>>>>>>>>>>>>>>>>>>>>>>>
        # self.num_view = 4
        # # (1, V, C)
        # self.view_token = nn.Parameter(torch.zeros(1, self.num_view, dim))
        self.pos_3d_linear = nn.Linear(3, dim)

        # >>>>>>>>>>>>>>>>>>>>>>>>> Variables >>>>>>>>>>>>>>>>>>>>>>>>>
        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")


        # >>>>>>>>>>>>>>>>>>>>>>>>> Joint Token (1, J, C) >>>>>>>>>>>>>>>>>>>>>>>>>
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = image_size[0] // (4*self.patch_size[0]), image_size[1] // (4* self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # >>>>>>>>>>>>>>>>>>>>>>>>> stem net >>>>>>>>>>>>>>>>>>>>>>>>> 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # >>>>>>>>>>>>>>>>>>>>>>>>> transformer >>>>>>>>>>>>>>>>>>>>>>>>> 
        self.transformer = TransformerMulti(dim, depth, heads, mlp_dim, dropout,num_keypoints=num_keypoints,all_attn=self.all_attn)

        self.to_keypoint_token = nn.Identity()

        # >>>>>>>>>>>>>>>>>>>>>>>>> Output Head >>>>>>>>>>>>>>>>>>>>>>>>> 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)
            
    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def stem_net_forward(self, img):
        # x: (B, 3, Hi, Wi)
        # >>>>>>>>>>>>>>>>>>>>>>>>> stem net >>>>>>>>>>>>>>>>>>>>>>>>> 
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)          # (B, C, H, W)  H = Hi/16

        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])        # flatten -> (B, HW, C)
        x = self.patch_to_embedding(x)      # (B, HW, C)
        return x      

    def single_view_forward(self, img, mask=None, ratio=0.7):
        # img: (B, 3, Hi, Wi)
        # >>>>>>>>>>>>>>>>>>>>>>>>> Stem Net >>>>>>>>>>>>>>>>>>>>>>>>> 
        x = self.stem_net_forward(img)      # (B, HW, C)
        
        # >>>>>>>>>>>>>>>>>>>>>>>>> single view transformer >>>>>>>>>>>>>>>>>>>>>>>>> 
        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :#
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)      # (B, J+HW, C)
        elif self.pos_embedding_type == "learnable":
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        x = self.transformer(x, mask,self.pos_embedding, prune=True, keep_ratio=ratio)      # (B, J+HW, C)

        # >>>>>>>>>>>>>>>>>>>>>>>>> output heatmap >>>>>>>>>>>>>>>>>>>>>>>>> 
        x = self.to_keypoint_token(x[:, 0:self.num_keypoints])          # (B, J, C)
        x = self.mlp_head(x)                                            # (B, J, HW)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        return x 

    def forward(self, imgs, mask = None, ratio=0.7, rays=None, centers=None, fuse=False):
        # >>>>>>>>>>>>>>>>>>>>>>>>> multi view >>>>>>>>>>>>>>>>>>>>>>>>> 
        if isinstance(imgs, list):
            # ******************* no fusion between each views ******************* 
            if not fuse:
                return [ self.single_view_forward(img, ratio=ratio) for img in imgs]
            
            # # ******************* no fusion between each views ******************* 
            else:
                xs, pos_embs  = [], []
                pos_embs_3d = []
                b = imgs[0].shape[0]
                keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)     # (B, J, C)
                for i, img in enumerate(imgs):
                    # >>>>>>>>>>>>>>>> backbone >>>>>>>>>>>>>>>>
                    x = self.stem_net_forward(img)      # (B, HW, C)
                    b, n, _ = x.shape

                    # >>>>>>>>>>>>>>>> 2D position encoding >>>>>>>>>>>>>>>>
                    pos_emb_2d = self.pos_embedding[:, :n].expand(b, -1, -1)    # (B, HW, C) only consider sine-full 2D PE here
                    # >>>>>>>>>>>>>>>> 3D position encoding >>>>>>>>>>>>>>>>
                    vec_c_p = F.normalize(rays[i] - centers[i], dim=2, p=2)     # (B, HW, 3)
                    pos_emb_3d = self.pos_3d_linear(vec_c_p)                    # (B, HW, 3) -> (B, HW, 256)
                    # pos_emb = pos_emb_2d + pos_emb_3d
                    pos_emb = pos_emb_2d

                    x += pos_emb

                    x = torch.cat((keypoint_tokens, x), dim=1)      # (B, J+HW, C)

                    xs.append(x)                       # (B, J+HW, C)
                    pos_embs.append(pos_emb)           # (B, hw, C)     2D only
                    pos_embs_3d.append(pos_emb_3d)     # (B, HW, C)     3D only
                
                xs = self.transformer(xs, mask, pos_embs, prune=True, keep_ratio=ratio, pos_emb_3ds=pos_embs_3d)      # list, (B, J+KV, C)


                outputs = []
                for x in xs:
                    # >>>>>>>>>>>>>>>>>>>>>>>>> output heatmap >>>>>>>>>>>>>>>>>>>>>>>>> 
                    x = self.to_keypoint_token(x[:, 0:self.num_keypoints])          # (B, J, C)
                    x = self.mlp_head(x)                                            # (B, J, HW)
                    x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
                    outputs.append(x)
                return outputs

        # >>>>>>>>>>>>>>>>>>>>>>>>> single view >>>>>>>>>>>>>>>>>>>>>>>>> 
        else:
            return self.single_view_forward(imgs, mask=mask, ratio=ratio)

        



# ******************************************************************************************
# ************************************** Token Pose Small **********************************
# ******************************************************************************************



class TokenPose_S(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(TokenPose_S, self).__init__()

        print(cfg.NETWORK)
        ##################################################
        self.features = TokenPose_S_base_multi(image_size=[cfg.NETWORK.IMAGE_SIZE[1],cfg.NETWORK.IMAGE_SIZE[0]],patch_size=[cfg.NETWORK.PATCH_SIZE[1],cfg.NETWORK.PATCH_SIZE[0]],
                                 num_keypoints = cfg.NETWORK.NUM_JOINTS,dim =cfg.NETWORK.DIM,
                                 channels=256,
                                 depth=cfg.NETWORK.TRANSFORMER_DEPTH,heads=cfg.NETWORK.TRANSFORMER_HEADS,
                                 mlp_dim = cfg.NETWORK.DIM*cfg.NETWORK.TRANSFORMER_MLP_RATIO,
                                 apply_init=cfg.NETWORK.INIT,
                                 hidden_heatmap_dim=cfg.NETWORK.HEATMAP_SIZE[1]*cfg.NETWORK.HEATMAP_SIZE[0]//8,
                                 heatmap_dim=cfg.NETWORK.HEATMAP_SIZE[1]*cfg.NETWORK.HEATMAP_SIZE[0],
                                 heatmap_size=[cfg.NETWORK.HEATMAP_SIZE[1],cfg.NETWORK.HEATMAP_SIZE[0]],
                                 pos_embedding_type=cfg.NETWORK.POS_EMBEDDING_TYPE)
        ###################################################3

    def forward(self, x, ratio=0.7, centers=None, rays=None, fuse=False):
        x = self.features(x, ratio=ratio, rays=rays, centers=centers, fuse=fuse)
        return x

    def init_weights(self, pretrained=''):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> from COCO pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
        if os.path.isfile(pretrained):
            logger.info('=> init final MLP head from normal distribution')
            for m in self.features.mlp_head.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained, map_location='cpu')
            logger.info('=> loading COCO Pretrained model {}'.format(pretrained))
            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name in self.state_dict():
                    #if 'mlp_head' in name or 'pos_embedding' in name or 'keypoint_token' in name or 'patch_to_embedding' in name:       # 2D Pos Embeddings
                    #    continue
                    if 'keypoint_token' in name:
                        new_m = torch.zeros(1, 17, 192)
                        # Human 36M -> MPII
                        # map_idx = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, 13, 14, 15, 12, 11, 10]
                        # Human 36M -> COCO
                        map_idx = [12, 12, 14, 16, 11, 13, 15, 11, 1, 0, 2, 5, 7, 9, 6, 8, 10]
                        new_m[0] = m[0][map_idx]
                        m = new_m
                        print('Shift Token ...')

                    existing_state_dict[name] = m
                    logger.info(":: {} is loaded from {}".format(name, pretrained))
                    print('Size: ', m.shape)

            self.load_state_dict(existing_state_dict, strict=False)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>> from scratch >>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)



def get_multiview_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_S(cfg, **kwargs)
    if is_train and cfg.NETWORK.INIT_WEIGHTS:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model






