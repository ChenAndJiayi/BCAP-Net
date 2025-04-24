
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import load_checkpoint
# from models.mit import mit_b4
from .rcf_encoder import dc_b1,dc_b3,dc_b5
import numpy as np
import torchvision.models as models

from torch.autograd import Function
from collections import OrderedDict

from timm.models.layers import DropPath
from torch.cuda.amp import custom_fwd, custom_bwd
import functools
from einops.layers.torch import Rearrange



class HMDA_decoder(nn.Module):
    def __init__(self):
        super(HMDA_decoder,self).__init__()
        
        self.conv_block4_1 = nn.Sequential(
                                           conv3x3(256, 64,'GN',32),
                                           conv3x3(64, 32,'GN',32//4))
        self.conv_block4_2 = nn.Sequential(conv3x3(32+1, 16,'GN',16//4),
                                           nn.Conv2d(16, 1,3,1,1) )
        
        self.conv_block3_1 = nn.Sequential(
                                           conv3x3(168, 64,'GN',32),
                                           conv3x3(64, 32,'GN',32//4))
        self.conv_block3_2 = nn.Sequential(conv3x3(32+2, 16,'GN',16//4),
                                           nn.Conv2d(16, 1,3,1,1) )
        
        self.conv_block2_1 = nn.Sequential(conv3x3(72, 32,'GN',32//4),
                                           conv3x3(32, 16,'GN',16//4))
        self.conv_block2_2 = nn.Sequential(conv3x3(16+2, 8,'GN',8//4),
                                           nn.Conv2d(8, 1,3,1,1) )
        
        self.conv_block1_1 = nn.Sequential(conv3x3(36, 16,'GN',16//4),
                                           conv3x3(16, 8,'GN',8//4))
        self.conv_block1_2 = nn.Sequential(conv3x3(8+2, 4,'GN',4//2),nn.Conv2d(4, 1,3,1,1))
        
        self.conv_block0_1 = nn.Sequential(conv3x3(16+16+8+1, 16,'GN',16//4),conv3x3(16, 8,'GN',8//4))
        self.conv_block0_2 = nn.Sequential(conv3x3(8+2, 4,'GN',4//2),nn.Conv2d(4, 1,3,1,1))
        self.upx2= nn.PixelShuffle(2)

        self.DCPR0 = DCPR(in_channels=8, pt=10,depth_num=5)
        self.DCPR1 = DCPR(in_channels=8, pt=10,depth_num=4)
        self.DCPR2 = DCPR(in_channels=16, pt=8,depth_num=3)
        self.DCPR3 = DCPR(in_channels=32, pt=6,depth_num=2)
        self.DCPR4 = DCPR(in_channels=32, pt=4,depth_num=1)

        self.wpool4 = WPool(32, level=4)
        self.wpool3 = WPool(32, level=3)
        self.wpool2 = WPool(16, level=2)
        self.wpool1 = WPool(8, level=1)
        self.max_depth = 80
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self,image_feats,sparse_feats,fuse_image,fuse_sparse,sparse_depth):
        depthF4 = self.upx2(torch.cat([fuse_image[3],fuse_sparse[3]],dim=1)) # 128*1/16
        depthF4 = self.conv_block4_1(depthF4) # 32
        SD4     = self.wpool4(sparse_depth,depthF4)
        depth4  = self.conv_block4_2(torch.cat([depthF4,SD4],dim=1))
        depth4  = self.DCPR4(depthF4,depth4,SD4)
        depth4  = F.interpolate(depth4, scale_factor=2, mode='bilinear', align_corners=True)
        
        depthF3 = self.upx2(torch.cat([fuse_image[2],fuse_sparse[2],depthF4],dim=1)) # 88*1/8
        depthF3 = self.conv_block3_1(depthF3) # 32
        SD3     = self.wpool3(sparse_depth,depthF3)
        depth3  = self.conv_block3_2(torch.cat([depthF3,depth4,SD3],dim=1))
        depths  = torch.cat([depth4,depth3],dim=1)
        depth3  = self.DCPR3(depthF3,depths,SD3)
        #--------
        depth4  = F.interpolate(depth4, scale_factor=2, mode='bilinear', align_corners=True)
        depth3  = F.interpolate(depth3, scale_factor=2, mode='bilinear', align_corners=True)
        
        depthF2 = self.upx2(torch.cat([fuse_image[1],fuse_sparse[1],depthF3],dim=1)) # 40*1/4
        depthF2 = self.conv_block2_1(depthF2) # 16
        SD2     = self.wpool2(sparse_depth,depthF2)
        depth2  = self.conv_block2_2(torch.cat([depthF2,depth3,SD2],dim=1))
        depths  = torch.cat([depth4,depth3,depth2],dim=1)
        depth2  = self.DCPR2(depthF2,depths,SD2)
        #----------
        depth4  = F.interpolate(depth4, scale_factor=2, mode='bilinear', align_corners=True)
        depth3  = F.interpolate(depth3, scale_factor=2, mode='bilinear', align_corners=True)
        depth2  = F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)
        
        depthF1 = self.upx2(torch.cat([fuse_image[0],fuse_sparse[0],depthF2],dim=1)) # 20*1/2
        depthF1 = self.conv_block1_1(depthF1) #8
        SD1     = self.wpool1(sparse_depth,depthF1)
        depth1  = self.conv_block1_2(torch.cat([depthF1,depth2,SD1],dim=1))
        depths  = torch.cat([depth4,depth3,depth2,depth1],dim=1)
        depth1  = self.DCPR1(depthF1,depths,SD1)
        #----
        depth4  = F.interpolate(depth4, scale_factor=2, mode='bilinear', align_corners=True)
        depth3  = F.interpolate(depth3, scale_factor=2, mode='bilinear', align_corners=True)
        depth2  = F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)
        depth1  = F.interpolate(depth1, scale_factor=2, mode='bilinear', align_corners=True)
        
        depthF1 = F.interpolate(depthF1, scale_factor=2, mode='bilinear', align_corners=True)
        depthF0 = self.conv_block0_1(torch.cat([self.upx2(image_feats[0]),self.upx2(sparse_feats[0]),depthF1,sparse_depth],dim=1))
        #-------------------
        depths  = torch.cat([depth4,depth3,depth2,depth1],dim=1)
        depth0  = self.DCPR0(depthF0,depths,sparse_depth)
        return depth4, depth3, depth2, depth1,depth0

class BCAPNet(nn.Module):
    def __init__(self, max_depth=10.0, mode='train'):
        super().__init__()
        self.max_depth = max_depth

        self.Sparse_encoder = dc_b5()
        self.decoder  = HMDA_decoder()
        
        self.downChn1 =  nn.Sequential(conv3x3(256, 128,'GN',32),conv3x3(128, 64,'GN',32))
        self.downChn2 =  nn.Sequential(conv3x3(512, 256,'GN',32),conv3x3(256, 128,'GN',32))
        self.downChn3 =  nn.Sequential(conv3x3(1024, 512,'GN',32),conv3x3(512, 320,'GN',32))
        self.downChn4 =  nn.Sequential(conv3x3(2048, 1024,'GN',32),conv3x3(1024, 512,'GN',32))
        if mode=='train':
            isPretrained =True
            ckpt_path = './models/weights/mit_b3.pth'
            load_checkpoint(self.Sparse_encoder, ckpt_path, logger=None)
        elif mode=='fine-t' or mode=='resume' or mode=='test':
            isPretrained = False
        else:
            print("error! need mode!")

        self.Image_encoder = deepFeatureExtractor_ResNext101(isPretrained=isPretrained, lv6=True)
        #
        num_params = sum([np.prod(p.size()) for p in self.Image_encoder.parameters()])
        print("Total number of encoder parameters: {}".format(num_params))
        num_params = sum([np.prod(p.size()) for p in self.Sparse_encoder.parameters()])
        print("Total number of encoder parameters: {}".format(num_params))
        num_params = sum([np.prod(p.size()) for p in self.decoder.parameters()])
        print("Total number of decoder parameters: {}".format(num_params))
        #
    def forward(self, image,sparse_depth):
        image_feats = self.Image_encoder(image)
        image_feats[1] = self.downChn1(image_feats[1])
        image_feats[2] = self.downChn2(image_feats[2])
        image_feats[3] = self.downChn3(image_feats[3])
        image_feats[4] = self.downChn4(image_feats[4])
        fuse_feats,sparse_feats = self.Sparse_encoder(sparse_depth,image_feats[1:])

        depth4, depth3, depth2, depth1,depth0 = self.decoder(image_feats,fuse_feats,sparse_feats,sparse_depth)

        return depth0, depth1, depth2, depth3,depth4
    def train(self, mode=True):
        super().train(mode)
        self.Image_encoder.freeze_bn()
