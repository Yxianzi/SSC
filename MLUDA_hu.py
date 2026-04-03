# -*- coding:utf-8 -*-
# Author：Mingshuo Cai (Modified for End-to-End EMA-SSC)
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset with End-to-End Spectral Style Calibration

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
from net2 import DSANSS
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_Houston import *
from sklearn import svm
from UtilsCMS import *


# ================= 纯 GPU 数据增强函数 =================
def radiation_noise_pt(data, alpha_range=(0.9, 1.1), beta=0.04):
    alpha = torch.empty(1, device=data.device).uniform_(*alpha_range)
    noise = torch.randn_like(data)
    return alpha * data + beta * noise


def flip_augmentation_pt(data):
    if torch.rand(1).item() > 0.5:
        data = torch.flip(data, dims=[-1])
    if torch.rand(1).item() > 0.5:
        data = torch.flip(data, dims=[-2])
    return data


# =======================================================

##################################
data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)

# 移除静态离线对齐
# data_s,data_t = ILDA(data_s,data_t,pca_n,radius)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    # 使用 from_numpy 保持原始高光谱数据
    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # 初始化主干网络与 SSC 校准网络
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
    ssc_module = SSC_Replacement(channels=nBand, r=1, eps=radius, momentum=0.1).cuda()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

        # 双优化器分离配置
        optimizer_backbone = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters()},
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        optimizer_ssc = torch.optim.Adam(ssc_module.parameters(), lr=1e-4, weight_decay=1e-4)

        feature_encoder.train()
        ssc_module.train()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader

        total_hit, size = 0.0, 0.0

        for i in range(1, num_iter):
            source_data, source_label = next(iter_source)
            target_data, target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            source_data_cuda = source_data.type(torch.FloatTensor).cuda()
            target_data_cuda = target_data.type(torch.FloatTensor).cuda()

            # ========== 端到端 EMA 光谱校准 ==========
            source_data_gf, source_calibrated = ssc_module(source_data_cuda, target_data_cuda)
            target_data_gf, _ = ssc_module(target_data_cuda, target_data_cuda)

            # ========== GPU 加速数据增强 ==========
            source_data0 = radiation_noise_pt(source_data_gf)
            source_data1 = flip_augmentation_pt(source_data_gf)
            target_data0 = radiation_noise_pt(target_data_gf)
            target_data1 = flip_augmentation_pt(target_data_gf)

            # 特征提取前向传播
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data_gf, target_data_gf)

            (_, source2, _, source_outputs2, _,
             _, _, target2, t1, _) = feature_encoder(source_data0, target_data0)

            (_, source3, _, source_outputs3, _,
             _, _, target3, t2, _) = feature_encoder(source_data1, target_data1)

            # ========== 损失计算 ==========
            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # ====== 新增: 更新全局类别原型 (EMA机制) ======
            feature_encoder.update_prototypes(
                f_s=source_features.detach(),
                label_s=source_label.cuda(),
                f_t=target_features.detach(),
                prob_t=softmax_output_t,
                threshold=0.9
            )

            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * epoch / epochs)) - 1
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)
            domain_similar_loss = DSH_loss(source_out, target_out)

            loss_backbone = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss

            # 物理一致性正则 (SAM Loss)，限制极端扭曲
            sam_loss = spectral_angle_loss(source_data_cuda, source_data_gf)
            sam_weight = max(0.05, 1.0 - 0.1 * epoch)  # 缓慢衰减至 0.05
            loss_ssc = sam_weight * sam_loss

            total_loss = loss_backbone + loss_ssc

            # 更新参数
            optimizer_backbone.zero_grad()
            optimizer_ssc.zero_grad()
            total_loss.backward()
            optimizer_backbone.step()
            optimizer_ssc.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

        test_accuracy = 100. * float(total_hit) / (size + 1e-8)

        print(
            'epoch {:>3d}: cls loss: {:6.4f}, lmmd loss: {:6.4f}, con_s loss: {:6.4f}, con_t loss: {:6.4f}, sam loss: {:6.4f}, acc {:6.4f}, total loss: {:6.4f}'
            .format(epoch, cls_loss.item(), lmmd_loss.item(), contrastive_loss_s.item(), contrastive_loss_t.item(),
                    sam_loss.item(), test_accuracy / 100.0, total_loss.item()))

        train_end = time.time()

        if epoch % epochs == 0:
            feature_encoder.eval()
            ssc_module.eval()  # 开启评估模式，触发 EMA 统计量推理

            total_rewards = 0
            counter = 0
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_datas_cuda = Variable(test_datas).type(torch.FloatTensor).cuda()

                    # 测试集通过 SSC 自校准（严格应用全局 EMA 统计量）
                    test_datas_gf, _ = ssc_module(test_datas_cuda, test_datas_cuda)

                    # 补齐 dummy_source 以满足网络双输入机制
                    curr_bs = test_datas_gf.shape[0]
                    dummy_source = source_data_gf[:curr_bs] if source_data_gf.shape[0] >= curr_bs else source_data_gf

                    source_features, source1, _, source_outputs, source_out, test_features, _, _, test_outputs, _ = feature_encoder(
                        dummy_source, test_datas_gf)

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            if test_accuracy > last_accuracy:
                print("save networks for epoch:", epoch)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe, last_accuracy))
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)

print("train time per DataSet(s): {:.5f}".format(train_end - train_start))
print("test time per DataSet(s): {:.5f}".format(test_end - train_end))
print("average OA: {:.2f} +- {:.2f}".format(OAMean, OAStd))
print("average AA: {:.2f} +- {:.2f}".format(100 * AAMean, 100 * AAStd))
print("average kappa: {:.4f} +- {:.4f}".format(100 * kMean, 100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class {}: {:.2f} +- {:.2f}".format(i, 100 * AMean[i], 100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

################# classification map ################################
for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1