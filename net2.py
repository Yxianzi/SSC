# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the Cross attention proposed in MLUDA.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 域专用批归一化 (Domain-Specific BN)
# ==========================================
class DSBN3d(nn.Module):
    def __init__(self, num_features):
        super(DSBN3d, self).__init__()
        self.bn_s = nn.BatchNorm3d(num_features)
        self.bn_t = nn.BatchNorm3d(num_features)

    def forward(self, x, domain='source'):
        if domain == 'source':
            return self.bn_s(x)
        elif domain == 'target':
            return self.bn_t(x)
        else:
            raise ValueError("Domain label must be 'source' or 'target'.")


# 2. 原型引导的交叉注意力 (Prototype-guided CA)
# ==========================================
class ProtoCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop_ratio=0.5, proj_drop_ratio=0.5):
        super(ProtoCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query 映射目标/源域 token，Key/Value 映射原型
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, query_feat, proto_feat):
        """
        query_feat: (B, D) 比如目标域的特征 token
        proto_feat: (C, D) 对应的语义原型 (C 为类别数)
        """
        B, D = query_feat.shape
        C, _ = proto_feat.shape

        x = self.norm1(query_feat)
        p = self.norm2(proto_feat)

        q = self.q_proj(x).reshape(B, self.num_heads, self.head_dim).permute(1, 0, 2)  # (H, B, d)
        k = self.k_proj(p).reshape(C, self.num_heads, self.head_dim).permute(1, 2, 0)  # (H, d, C)
        v = self.v_proj(p).reshape(C, self.num_heads, self.head_dim).permute(1, 0, 2)  # (H, C, d)

        attn = (q @ k) * self.scale  # (H, B, C)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (H, B, d)
        out = out.permute(1, 0, 2).reshape(B, D)  # (B, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        # 引入残差使得特征融合平滑
        return query_feat + out

class DSANSS(nn.Module):
    def __init__(self, n_band=198, patch_size=3, num_class=7, momentum=0.9):
        super(DSANSS, self).__init__()
        self.num_class = num_class
        self.momentum = momentum
        self.n_outputs = 288

        self.feature_layers = DCRN_02_DSBN(n_band, patch_size, num_class)

        # 类别原型缓冲 (C, D)
        self.register_buffer('source_prototypes', torch.zeros(num_class, self.n_outputs))
        self.register_buffer('target_prototypes', torch.zeros(num_class, self.n_outputs))

        # 原型引导的交叉注意力机制
        self.proto_attn_t2s = ProtoCrossAttention(dim=self.n_outputs)
        self.proto_attn_s2t = ProtoCrossAttention(dim=self.n_outputs)

        self.fc1 = nn.Linear(self.n_outputs, num_class)
        self.fc2 = nn.Linear(self.n_outputs, 1)

        self.head1 = nn.Sequential(nn.Linear(self.n_outputs, 128))
        self.head2 = nn.Sequential(nn.Linear(self.n_outputs, 128))
        self.sigmoid = nn.Sigmoid()

    def update_prototypes(self, f_s, label_s, f_t, prob_t, threshold=0.9):
        """外部调用：基于批次特征更新全局 EMA 原型"""
        # 更新源域原型
        for c in range(self.num_class):
            mask_s = (label_s == c)
            if mask_s.sum() > 0:
                curr_p_s = f_s[mask_s].mean(dim=0)
                if self.source_prototypes[c].sum() == 0:
                    self.source_prototypes[c] = curr_p_s.detach()
                else:
                    self.source_prototypes[c] = self.momentum * self.source_prototypes[c] + (
                                1 - self.momentum) * curr_p_s.detach()

        # 更新目标域原型 (利用高置信度伪标签)
        max_prob, pseudo_label = torch.max(prob_t, dim=1)
        for c in range(self.num_class):
            mask_t = (pseudo_label == c) & (max_prob > threshold)
            if mask_t.sum() > 0:
                curr_p_t = f_t[mask_t].mean(dim=0)
                if self.target_prototypes[c].sum() == 0:
                    self.target_prototypes[c] = curr_p_t.detach()
                else:
                    self.target_prototypes[c] = self.momentum * self.target_prototypes[c] + (
                                1 - self.momentum) * curr_p_t.detach()

    def forward(self, x, y):
        # 1. 经过 DSBN 分离归一化
        features_x, features_y = self.feature_layers(x, y)

        # 2. 原型引导的语义对齐 (如果原型已经初始化)
        if self.source_prototypes.sum() != 0:
            aligned_y = self.proto_attn_t2s(features_y, self.source_prototypes)
        else:
            aligned_y = features_y

        if self.target_prototypes.sum() != 0:
            aligned_x = self.proto_attn_s2t(features_x, self.target_prototypes)
        else:
            aligned_x = features_x

        # 后续投影和分类
        x1 = F.normalize(self.head1(aligned_x), dim=1)
        x2 = F.normalize(self.head2(aligned_x), dim=1)
        fea_x = self.fc1(aligned_x)
        output_x = self.sigmoid(self.fc2(aligned_x))

        y1 = F.normalize(self.head1(aligned_y), dim=1)
        y2 = F.normalize(self.head2(aligned_y), dim=1)
        fea_y = self.fc1(aligned_y)
        output_y = self.sigmoid(self.fc2(aligned_y))

        return aligned_x, x1, x2, fea_x, output_x, aligned_y, y1, y2, fea_y, output_y

    def get_embedding(self, x):
        # 单独推理时作为 target
        feat = self.feature_layers.forward_single_domain(x, domain='target')
        if self.source_prototypes.sum() != 0:
            feat = self.proto_attn_t2s(feat, self.source_prototypes)
        return feat

class DSAN1(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 288
        self.feature_layers = DCRN_02(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )

        self.head2 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN2(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 152
        self.feature_layers = DCRN(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(self.n_outputs, num_class)
        self.fc2 = nn.Linear(self.n_outputs, 1)

        self.head1 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out


# 3. 包含 DSBN 的骨干网络 (替换原 DCRN_02)
# ==========================================
class DCRN_02_DSBN(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN_02_DSBN, self).__init__()
        self.feature_dim = input_channels
        self.sz = patch_size

        # 替换所有的 BatchNorm3d 为 DSBN3d
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = DSBN3d(24)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn2 = DSBN3d(24)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn3 = DSBN3d(24)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = DSBN3d(192)
        self.activation4 = nn.ReLU()

        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = DSBN3d(24)
        self.activation5 = nn.ReLU()

        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn6 = DSBN3d(24)
        self.activation6 = nn.ReLU()

        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn7 = DSBN3d(96)
        self.activation7 = nn.ReLU()

        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)

        self.inter_size = 192 + 96

        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)
        self.bn9 = DSBN3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)
        self.bn10 = DSBN3d(self.inter_size)
        self.activation10 = nn.ReLU()

        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward_single_domain(self, x, domain):
        """处理单域前向传播，明确区分统计量"""
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1, domain))

        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1, domain))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1, domain))

        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1, domain))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))

        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2, domain))

        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2, domain))
        x2 = self.conv7(x2)
        x2 = residual + x2
        x2 = self.activation7(self.bn7(x2, domain))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))

        out = torch.cat((x1, x2), 1)
        ca_out = self.ca(out)
        sa_out = self.sa(out)
        out = ca_out * out
        out = sa_out * out

        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        return out

    def forward(self, x, y):
        feat_x = self.forward_single_domain(x, domain='source')
        feat_y = self.forward_single_domain(y, domain='target')
        return feat_x, feat_y

class CrossAttention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,   
                 qkv_bias=False,    
                 qk_scale=None,     
                 attn_drop_ratio=0.,  
                 proj_drop_ratio=0.): 

        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads    
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_before = nn.Linear(head_dim,head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.atten_norm = nn.LayerNorm(dim,eps=1e-6)
        self.atten_norm_before = nn.LayerNorm(head_dim, eps=1e-6)
    def forward(self, x,y,inial_y2x,inial_x2y):
        x = self.atten_norm(x)
        y = self.atten_norm(y)
        inial_y2x = self.atten_norm(inial_y2x)
        inial_x2y = self.atten_norm(inial_x2y)

        B_x, C_x = x.shape
        B_y, C_y = y.shape

        qkv_x = self.qkv(x).reshape(B_x, 3, self.num_heads, C_x // self.num_heads).permute(1 ,0 ,2 ,3)
        qkv_y = self.qkv(y).reshape(B_y, 3, self.num_heads, C_y // self.num_heads).permute(1, 0, 2, 3)
        q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]  # make torchscript happy (cannot use tensor as tuple)
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2]  # make torchscript happy (cannot use tensor as tuple)

        attn_x = torch.mul(q_x,k_x) * self.scale  
        attn_x = attn_x.softmax(dim=-1)  
        attn_x = self.attn_drop(attn_x) 
        F_x = torch.mul(attn_x,v_x)
        F_x = self.proj_before(F_x)
        F_x = F_x.reshape(B_x,C_x)
        F_x = F_x + x
        F_x = self.atten_norm(F_x)
        Final_x = self.proj(F_x)
        Final_x = self.proj_drop(Final_x)
        Final_x = Final_x + F_x   

        attn_y = torch.mul(q_y,k_y) * self.scale  
        attn_y = attn_y.softmax(dim=-1)  
        attn_y = self.attn_drop(attn_y)  
        F_y = torch.mul(attn_y,v_y)
        F_y = self.proj_before(F_y)
        F_y = F_y.reshape(B_y,C_y)
        F_y = F_y + y
        F_y = self.atten_norm(F_y)
        Final_y = self.proj(F_y)
        Final_y = self.proj_drop(Final_y)
        Final_y = Final_y + F_y   


        attn_y2x = torch.mul(q_y,k_x) * self.scale  
        attn_y2x = attn_y2x.softmax(dim=-1)    
        attn_y2x = self.attn_drop(attn_y2x)    
        y2x = torch.mul(attn_y2x,v_x)
        y2x = self.proj_before(y2x)
        y2x = y2x.reshape(B_x,C_x)
        y2x = y2x + inial_y2x
        y2x = self.atten_norm(y2x)
        Final_y2x = self.proj(y2x)
        Final_y2x = self.proj_drop(Final_y2x)
        Final_y2x = Final_y2x + y2x  

        attn_x2y = torch.mul(q_x,k_y) * self.scale   
        attn_x2y = attn_x2y.softmax(dim=-1)    
        attn_x2y = self.attn_drop(attn_x2y)    
        x2y = torch.mul(attn_x2y, v_y)
        x2y = self.proj_before(x2y)
        x2y = x2y.reshape(B_x, C_x)
        x2y = x2y + inial_x2y
        x2y = self.atten_norm(x2y)
        Final_x2y = self.proj(x2y)
        Final_x2y = self.proj_drop(Final_x2y)
        Final_x2y = Final_x2y + x2y  

        return Final_x,Final_y,Final_y2x,Final_x2y
class DCRN(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN, self).__init__()
        self.kernel_dim = 1                 
        self.feature_dim = input_channels  
        self.sz = patch_size                

        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)#padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),bias=True)# padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),bias=True)#padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)#self.inter_size
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # 定义参数的初始化形式
        for m in self.modules():
            if isinstance(m, nn.Conv3d):    
                torch.nn.init.kaiming_normal_(m.weight.data)   
                m.bias.data.zero_()                         
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:                
            nn.init.xavier_uniform_(m.weight, gain=1)   
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = torch.ones(m.bias.data.size())



    def forward(self, x,y):

        x = x.unsqueeze(1) # (64,1,100,9,9)  -> (64,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1                                                                                                                                                                                                                                                                                                                                
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1                  #(32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)                 #(32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4)) #(32,128,7,7)

        x2 = self.conv5(x)                      #(32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)     #(32,24,1,7,7)
        x2 = self.conv6(x2)                 #(32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)                 #(32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) #(32,24,7,7)

        x = torch.cat((x1, x2), 1)      #(32,152,7,7)


        ###################
        # attention map
        ###################

        x = self.ca(x) * x                  
        x = self.sa(x) * x                  

        x = self.avgpool(x)                 
        x = x.view(x.shape[0], -1) 

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

