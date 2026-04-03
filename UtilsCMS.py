# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the GuidedPGC proposed in MLUDA.

import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
# from config_Houston import *
from sklearn import svm
import cv2
import hdf5storage
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
# 自己加入的库
from sklearn.decomposition import PCA
from skimage import exposure
import argparse
import cv2

def PlotColor(label):
    "将Label以热力图的形式进行可视化"
    plt.imshow(label, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.title('Source')
    plt.show()

def pca(data,n):
    "进行pca过程的代码"
    pca = PCA(n_components=n)   # 定义pca的方法
    height, width, channels = data.shape
    data = data.reshape(-1,channels)
    data_PCA = pca.fit_transform(data).reshape(height,width,n)
    Score = pca.explained_variance_ratio_
    # 检验三个主成分是否可以代表
    # print("前三个主成分的贡献程度:")
    # print(Score)
    # 在每个通道上进行归一化操作    有没有可能这里的归一化会把东西搞坏
    min_value = np.min(data_PCA, axis=(0, 1))  # 在前两个维度上找最小值
    max_value = np.max(data_PCA, axis=(0, 1))  # 在前两个维度上找最大值
    data_PCA = (data_PCA - min_value) / (max_value - min_value)
    return data_PCA,Score

def gammaCorrect(SourcePic,TargetPic):
    """
    进行gamma校准的代码
    SourcePic:一个传入源域的三维的图像
    TargetPic:一个传入目标的三维的图像
    """
    # 分别计算源域和目标域的亮度
    # Sourcebrightness1 = 0.299 * SourcePic[:,:,0] + 0.587 * SourcePic[:,:,1] + 0.114 * SourcePic[:,:,2]
    # Targetbrightness1 = 0.299 * TargetPic[:,:,0] + 0.587 * TargetPic[:,:,1] + 0.114 * TargetPic[:,:,2]
    Sourcebrightness1 = np.mean(SourcePic)
    Targetbrightness1 = np.mean(TargetPic)
    # 计算需要应用的Gamma值，以使两个图像的亮度一致
    average_brightness_Source = np.mean(Sourcebrightness1)
    average_brightness_Target = np.mean(Targetbrightness1)
    gamma_value = abs(average_brightness_Source / average_brightness_Target )
    # 对源域和目标域的图像分别应用gamma校准
    Source_corrected = np.power(SourcePic,gamma_value)
    Target_corrected = TargetPic
    Target_corrected = np.clip(Target_corrected, 0, 1)
    # Sourcebrightness2 = (0.299 * Source_corrected[:,:,0]
    #                      + 0.587 * Source_corrected[:,:,1] + 0.114 * Source_corrected[:,:,2])
    # Targetbrightness2 = (0.299 * Target_corrected[:,:,0]
    #                      + 0.587 * Target_corrected[:,:,1] + 0.114 * Target_corrected[:,:,2])
    Sourcebrightness2 = np.mean(Source_corrected)
    Targetbrightness2 = np.mean(Target_corrected)
    print("gamma:"+str(gamma_value))
    print(Sourcebrightness1.mean())
    print(Targetbrightness1.mean())
    print(Sourcebrightness2.mean())
    print(Targetbrightness2.mean())
    Brightness = [Sourcebrightness1.sum(),Targetbrightness1.sum(),
                  Sourcebrightness2.sum(),Targetbrightness2.sum()]
    return Source_corrected,Target_corrected,Brightness

def colorAdaption(Source_image,Target_image):
    # 进行颜色直方图匹配的代码
    Source_image_Matched = exposure.match_histograms(Source_image, Target_image)
    return Source_image_Matched,Target_image

def GuideFilter(Pic_MChannels, Pic_3Channels,r):
    # 进行导向滤波的代码
    Pic_3Channels = Pic_3Channels.astype(np.float32)
    Pic_MChannels = Pic_MChannels.astype(np.float32)
    # 执行导向滤波
    print("1, " + str(r))
    radius = 1  # 滤波半径
    epsilon =  r  # 正则化参数
    filtered_image = cv2.ximgproc.guidedFilter(Pic_3Channels,Pic_MChannels, radius, epsilon)
    return filtered_image


def TotalAdaption(Source_Pic, Target_Pic, pca_n,r):
    """
    进行图像级域迁移的代码
    """
    Source_data, Source_pca_score = pca(Source_Pic, pca_n)
    Target_data, Target_pca_score = pca(Target_Pic, pca_n)
    # Pca之前的操作都是没有问题的
    Source_data_gamma, Target_data_gamma, Brightness = gammaCorrect(Source_data, Target_data)  # 经过gamma校准
    print(Source_data_gamma.shape,Target_data_gamma.shape)
    Source_data_Color, Target_data_Color = colorAdaption(Source_data_gamma, Target_data_gamma)  # 经过颜色直方图匹配
    print(Source_data_Color.shape,Target_data_Color.shape)
    # print("经过gamma校准和颜色直方图匹配之后的效果展示：")
    # fig, axes = plt.subplots(3, 2, figsize=(90, 10))
    # axes[0, 0].imshow(Source_data)
    # axes[0, 0].set_title('源域经过PCA之后的三通道,亮度为' + str(Brightness[0]))  # 设置子图标题
    # axes[0, 1].imshow(Target_data)
    # axes[0, 1].set_title('目标域经过PCA之后的三通道,亮度为' + str(Brightness[1]))  # 设置子图标题
    # axes[1, 0].imshow(Source_data_gamma)
    # axes[1, 0].set_title('源域经过PCA，gamma校准之后的三通道,亮度为' + str(Brightness[2]))  # 设置子图标题
    # axes[1, 1].imshow(Target_data_gamma)
    # axes[1, 1].set_title('目标域经过PCA,gamma校准之后的三通道,亮度为' + str(Brightness[3]))  # 设置子图标题
    # axes[2, 0].imshow(Source_data_Color)
    # axes[2, 0].set_title('源域经过颜色直方图匹配之后的三通道:')  # 设置子图标题
    # axes[2, 1].imshow(Target_data_Color)
    # axes[2, 1].set_title('目标域经过颜色直方图之后的三通道:')  # 设置子图标题
    # plt.savefig('PIC/Gamma.png', dpi=300)
    Final_Source = GuideFilter(Source_Pic, Source_data_Color,r)
    Final_Target = GuideFilter(Target_Pic, Target_data_Color,r)

    return Final_Source, Final_Target

def ILDA(data_s,data_t,pca_n,r):
    # if param is None:
    #     # Pavia数据集
    #     # data_path_s = './datasets/Pavia/paviaU.mat'
    #     # label_path_s = './datasets/Pavia/paviaU_gt_7.mat'
    #     # data_path_t = './datasets/Pavia/pavia.mat'
    #     # label_path_t = './datasets/Pavia/pavia_gt_7.mat'
    #     # Houston数据集
    #     data_path_s = './datasets/Houston/Houston13.mat'
    #     label_path_s = './datasets/Houston/Houston13_7gt.mat'
    #     data_path_t = './datasets/Houston/Houston18.mat'
    #     label_path_t = './datasets/Houston/Houston18_7gt.mat'
    #     data_s,label_s = load_data_houston(data_path_s,label_path_s)
    #     data_t,label_t = load_data_houston(data_path_t,label_path_t)
    # 图像级迁移的执行部分
    ILDA_S,ILDA_T= TotalAdaption(data_s,data_t,pca_n,r)
    return ILDA_S,ILDA_T


class DiffGuidedFilter(nn.Module):
    """可微导向滤波 (数值稳定版)"""

    def __init__(self, r=1, eps=1e-8):
        super(DiffGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(kernel_size=2 * r + 1, stride=1, padding=r, count_include_pad=False)

    def forward(self, guidance, src):
        N_x = self.boxfilter(torch.ones_like(guidance))
        mean_x = self.boxfilter(guidance) / (N_x + 1e-8)
        mean_y = self.boxfilter(src) / (N_x + 1e-8)
        mean_xx = self.boxfilter(guidance * guidance) / (N_x + 1e-8)
        mean_xy = self.boxfilter(guidance * src) / (N_x + 1e-8)

        # 核心修复：增加 torch.clamp 强制方差 >= 0，防止浮点误差导致负数
        var_x = torch.clamp(mean_xx - mean_x * mean_x, min=0.0)
        cov_xy = mean_xy - mean_x * mean_y

        a = cov_xy / (var_x + self.eps)
        b = mean_y - a * mean_x

        mean_a = self.boxfilter(a) / (N_x + 1e-8)
        mean_b = self.boxfilter(b) / (N_x + 1e-8)

        return mean_a * guidance + mean_b


class SSC_Replacement(nn.Module):
    """端到端光谱风格校准模块"""

    def __init__(self, channels, r=1, eps=0.009):
        super(SSC_Replacement, self).__init__()
        self.stats_encoder = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU()
        )
        self.fc_alpha = nn.Linear(channels, channels)
        self.fc_beta = nn.Linear(channels, channels)
        self.gf = DiffGuidedFilter(r=r, eps=eps)

        self._initialize_identity()

    def _initialize_identity(self):
        """近似恒等映射初始化"""
        nn.init.zeros_(self.fc_alpha.weight)
        nn.init.zeros_(self.fc_alpha.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x_s, x_t):
        B, C, H, W = x_s.shape
        t_mean = x_t.view(B, C, -1).mean(dim=2)
        t_var = x_t.view(B, C, -1).var(dim=2)
        t_stats = torch.cat([t_mean, t_var], dim=1)

        hidden = self.stats_encoder(t_stats)
        delta_alpha = self.fc_alpha(hidden).view(B, C, 1, 1)
        beta = self.fc_beta(hidden).view(B, C, 1, 1)

        alpha = 1.0 + delta_alpha
        x_calibrated = alpha * x_s + beta
        x_gf = self.gf(guidance=x_calibrated, src=x_s)
        return x_gf, x_calibrated


def spectral_angle_loss(x, x_calibrated):
    """物理一致性约束：SAM Loss (数值稳定版)"""
    # 直接在空间维度上逐像素计算，避免 view 带来的维度混淆
    dot_product = torch.sum(x * x_calibrated, dim=1)

    # 手动计算 L2 范数，并在根号内 + 1e-8，彻底解决全 0 Padding 导致导数为 NaN 的问题
    norm_x = torch.sqrt(torch.sum(x ** 2, dim=1) + 1e-8)
    norm_xc = torch.sqrt(torch.sum(x_calibrated ** 2, dim=1) + 1e-8)

    cos_theta = dot_product / (norm_x * norm_xc + 1e-8)
    # 稍微收紧 clamp 的范围，避免 acos 在极端接近 1 或 -1 时梯度不稳定
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

    return torch.mean(torch.acos(cos_theta))

