# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset

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
from ssc_module import EndToEndSSC

##################################
data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s,label_s = utils.load_data_houston(data_path_s,label_path_s)
data_t,label_t = utils.load_data_houston(data_path_t,label_path_t)

"""data_s,data_t = ILDA(data_s,data_t,pca_n,radius)"""
feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
ssc_module = SSC_Replacement(channels=nBand, r=1, eps=radius).cuda()

ssc_warmup_epochs = 5

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
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    # 修改后 (使用 from_numpy)
    train_dataset = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY).long())
    test_dataset = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY).long())

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []

    train_start = time.time()

    #loss plot
    loss1 = []
    loss2 = []
    loss3 = []


    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

        # 优化器分离
        optimizer_backbone = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters()},
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        # 优化器 2：SSC 模块（使用更小的学习率，如主干的 0.1 倍，防止破坏图像结构）
        optimizer_ssc = torch.optim.Adam(ssc_module.parameters(), lr=lr * 0.1, weight_decay=1e-4)

        # 模块状态设置：直接进行联合训练
        feature_encoder.train()
        ssc_module.train()

        # 状态控制
        if epoch <= ssc_warmup_epochs:
            feature_encoder.eval()
            ssc_module.train()
            for param in feature_encoder.parameters(): param.requires_grad = False
            for param in ssc_module.parameters(): param.requires_grad = True
        else:
            feature_encoder.train()
            ssc_module.train()
            for param in feature_encoder.parameters(): param.requires_grad = True
            for param in ssc_module.parameters(): param.requires_grad = True

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader

        for i in range(1,num_iter):
            source_data, source_label = next(iter_source)
            target_data, target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            source_data_cuda = source_data.cuda()
            target_data_cuda = target_data.cuda()

            # --- 核心改动 1：原始数据输入 SSC 模块 ---
            # 目标域引导源域进行校准，输出导向滤波后的源域图像和用于计算 SAM loss 的纯校准图像
            source_data_gf, source_calibrated = ssc_module(source_data_cuda, target_data_cuda)
            # 目标域进行自校准（恒等映射近似）
            target_data_gf, _ = ssc_module(target_data_cuda, target_data_cuda)

            # --- 核心改动 2：基于 SSC 输出进行数据增强 (需使用 GPU 版本的增广函数) ---
            source_data0 = utils.radiation_noise_pt(source_data_gf)
            source_data1 = utils.flip_augmentation_pt(source_data_gf)
            target_data0 = utils.radiation_noise_pt(target_data_gf)
            target_data1 = utils.flip_augmentation_pt(target_data_gf)

            # --- 核心改动 3：将 SSC 输出送入特征提取器 ---
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(source_data_gf, target_data_gf)

            (_, source2, _, source_outputs2, _,
             _, _, target2, t1, _) = feature_encoder(source_data0, target_data0)

            (_, source3, _, source_outputs3, _,
             _, _, target3, t2, _) = feature_encoder(source_data1, target_data1)

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)],dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            # Loss Lmmd
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            # Loss Con_s
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            # Loss Con_t
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)
            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)

            # 1. 计算原 MLUDA 损失总和
            loss_backbone = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss

            # 2. 计算 SSC 物理约束损失 (约束源域原始图像与 SSC 仿射输出的光谱角不发生畸变)
            sam_loss = spectral_angle_loss(source_data_cuda, source_calibrated)

            # 3. 损失合并
            total_loss = loss_backbone + 1.0 * sam_loss

            # 4. 联合更新参数
            optimizer_backbone.zero_grad()
            optimizer_ssc.zero_grad()

            total_loss.backward()

            optimizer_backbone.step()
            optimizer_ssc.step()

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

            test_accuracy = 100. * float(total_hit) / size

            # 兼容预热期：如果还在预热期，主干网络的损失赋值为 0.0
            c_loss = cls_loss.item() if epoch > ssc_warmup_epochs else 0.0
            l_loss = lmmd_loss.item() if epoch > ssc_warmup_epochs else 0.0
            cs_loss = contrastive_loss_s.item() if epoch > ssc_warmup_epochs else 0.0
            ct_loss = contrastive_loss_t.item() if epoch > ssc_warmup_epochs else 0.0

            print(
                'epoch {:>3d}: cls loss: {:6.4f}, lmmd loss: {:6.4f}, con_s loss: {:6.4f}, con_t loss: {:6.4f}, sam loss: {:6.4f}, acc {:6.4f}, total loss: {:6.4f}'
                .format(epoch,
                        c_loss,
                        l_loss,
                        cs_loss,
                        ct_loss,
                        sam_loss.item(),
                        total_hit / (size + 1e-8),  # 防止 size 为 0 除零报错
                        total_loss.item()))

        train_end = time.time()
        if epoch % epochs == 0:
            # print("Testing ...")
            feature_encoder.eval()
            ssc_module.eval()

            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_datas_cuda = Variable(test_datas).type(torch.FloatTensor).cuda()

                    # --- 测试时目标域数据经过 SSC 自校准 ---
                    test_datas_gf, _ = ssc_module(test_datas_cuda, test_datas_cuda)

                    # 为了满足 feature_encoder 双输入的要求，构造一个与测试批次相同大小的 dummy_source
                    curr_bs = test_datas_cuda.shape[0]
                    dummy_source = source_data_gf[:curr_bs] if source_data_gf.shape[0] >= curr_bs else source_data_gf

                    # 送入特征提取器
                    source_features, source1, _, source_outputs, source_out, test_features, _, _, test_outputs, _ = feature_encoder(
                        dummy_source, test_datas_gf)

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            OA = acc
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode

            if test_accuracy > last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
                print("save networks for epoch:", epoch + 1)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/housotn18.png")
