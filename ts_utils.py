import random
from collections import Counter
import os
import numpy as np
import torch
import torch.optim
import shutil
import sys
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
# from ts_data.preprocessing import load_data_UCR, transfer_labels, k_fold, get_category_list
from ts_data.data_load import *
from ts_model.loss_3_9 import cross_entropy, reconstruction_loss
from ts_model.model import FCN, Classifier, BiLSTM, DilatedCNN, FCN_prj, FCN_expert

import logging
from datetime import datetime

def label_propagation(train_x_allq, train_y_allq, train_mask_allq, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.95, num_real_class=2, num_iterations=1, logger=None, epsilon=0.1, **kwargs):
    '''
    标签从已知的数据点“传播”到未知的数据点。
    对于未标记的样本（即那些在mask_label中标记为1的样本），它们的标签是由它们在KNN图中的邻居的已知标签共同决定的。
    '''


    data_embed = torch.cat([train_x_allq[j] for j in range(len(train_x_allq))], 0)
    y_label = torch.cat([train_y_allq[j] for j in range(len(train_y_allq))], 0)
    mask_label = np.concatenate(train_mask_allq)
    
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d

    # Calculate the weighted euclidean distance
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    # Adjust weights for class balance
    # Compute class weights inversely proportional to class frequencies
    class_weights = torch.zeros(num_real_class).to(device)
    for i in range(num_real_class):
        class_weights[i] = 1. / (y_label == i).sum().float()

    # Apply class weights to the distance matrix
    for i in range(n):
        if mask_label[i] == 0:  # Apply weights only for labeled data
            w[i] *= class_weights[int(y_label[i])]

    # keep top-k values
    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # Ensure the graph is symmetrical
    w = w * mask

    # normalize
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    # step2: label propagation with label smoothing
    y = torch.zeros(y_label.shape[0], num_real_class).to(device)    # label initialization
    y.fill_(epsilon / (num_real_class - 1))  # Label smoothing
    for i in range(n):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1 - epsilon

    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y)  # Label propagation

    all_knn_label = torch.argmax(f, 1).cpu().numpy()    # 选择标签传播结果中概率最高的标签作为最终标签
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    # class_counter = Counter(y_label)
    class_counter = [0] * num_real_class
    
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i]
        else:
            class_counter[all_knn_label[i]] += 1
    
    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        # classwise_acc[i] = class_counter[i] / max(class_counter.values())
        classwise_acc[i] = class_counter[i] / max(class_counter)
    
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    # 对于准确率较高的类别，阈值会降低，而对于准确率较低的类别，阈值则相对较高。
    # 比较max_probs中的每个元素是否大于或等于这个调整后的阈值。
    # 结果是一个布尔张量cpl_mask，其元素值为True表示对应的预测概率满足条件，可以被用于生成伪标签
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))    # .ge: greater than or equal to

    return all_knn_label, end_knn_label, cpl_mask, indices

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

def build_dataset(args, logger):    # 2.23

    function_name = f"load_{args.dataset}"

    try:
        if args.dataset in [
            'AllGestureWiimoteX',
            'AllGestureWiimoteY',
            'AllGestureWiimoteZ',
            'ArrowHead',
            'BME',
            'Car',
            'CBF',
            'Chinatown',
            'ChlorineConcentration',
            'CinCECGTorso',
            'Computers',
            'CricketX',
            'CricketY',
            'CricketZ',
            'Crop',
            'DiatomSizeReduction',
            'DistalPhalanxOutlineAgeGroup',
            'DistalPhalanxOutlineCorrect',
            'DistalPhalanxTW',
            'DodgerLoopGame',
            'DodgerLoopWeekend',
            'Earthquakes',
            'ECG200',
            'ECG5000',
            'ECGFiveDays',
            'ElectricDevices',
            'EOGHorizontalSignal',
            'EOGVerticalSignal',
            'EthanolLevel',
            'FaceAll',
            'FacesUCR',
            'Fish',
            'FordA',
            'FordB',
            'Fungi',
            'FreezerRegularTrain',
            'FreezerSmallTrain',
            'GestureMidAirD1',
            'GestureMidAirD2',
            'GestureMidAirD3',
            'GesturePebbleZ1',
            'GesturePebbleZ2',
            'GunPoint',
            'GunPointAgeSpan',
            'GunPointMaleVersusFemale',
            'GunPointOldVersusYoung',
            'Ham',
            'HandOutlines',
            'Haptics',
            'Herring',
            'HouseTwenty',
            'InsectEPGRegularTrain',
            'InsectEPGSmallTrain',
            'MelbournePedestrian',
            'PickupGestureWiimoteZ',
            'PigAirwayPressure',
            'PigArtPressure',
            'PigCVP',
            'PLAID',
            'PowerCons',
            'Rock',
            'SemgHandGenderCh2',
            'SemgHandMovementCh2',
            'SemgHandSubjectCh2',
            'ShakeGestureWiimoteZ',
            'SmoothSubspace',
            'UMD',
            'UWaveGestureLibraryAll',
            'UWaveGestureLibraryX',
            'UWaveGestureLibraryY',
            'UWaveGestureLibraryZ',
            'Wafer',
            'Wine',
            'WordSynonyms',
            'Worms',
            'WormsTwoClass',
            'Yoga',
        ]:
            sum_dataset, sum_target, num_classes, num_class_list = load_data_UCR(args.dataroot, args.dataset)
            sum_target = transfer_labels(sum_target)
            
        elif args.dataset in [
            'ECG',
            'EMG',
            'FDA',
            'FDB',
            'Gesture',
            'HAR',
            'SleepEDF',
            'SleepEEG',
        ]:
            sum_dataset, sum_target, num_classes, num_class_list = load_data_pt(args.dataroot, args.dataset)
            sum_target = transfer_labels(sum_target)

        else:                  
            # 使用getattr动态获取对应的函数
            loader_function = getattr(sys.modules[__name__], function_name)
            # 调用函数并返回结果
            sum_dataset, sum_target, num_classes, num_class_list = loader_function(args.dataroot, args.dataset)
            sum_target = transfer_labels(sum_target)

        args.seq_len = sum_dataset.shape[1]
   
        # 如果当前样本数量的60%少于当前 batch_size，则设置batch_size减半
        while sum_dataset.shape[0] * 0.6 < args.batch_size:
            args.batch_size = args.batch_size // 2

        # 如果调整后的批处理大小的两倍大于数据集总量的60%，那么将args.queue_maxsize设置为2
        if args.batch_size * 2 > sum_dataset.shape[0] * 0.6:
            logger.info('queue_maxsize is changed to 2')
            args.queue_maxsize = 2

        train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
            sum_dataset, sum_target)    # 进入 preprocessing 的 k_fold
        
        return train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list
    
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
        raise ValueError(f"未找到名为'{function_name}'的数据集加载函数。")
    
def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool
    
    if args.backbone == 'fcn_prj':
        model = FCN_prj(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool

    if args.backbone == 'fcn_expert':
            model = FCN_expert(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool
        

    if args.backbone == 'lstm':
        model = BiLSTM(args.num_classes, args.input_size)  
    
    if args.backbone == 'dilated':
        model = DilatedCNN(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool

    if args.classifier == 'linear':
        classifier = Classifier(args.classifier_input, args.num_classes)    # linear classifier

    return model, classifier

def build_experts_in_on_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool
    
    if args.backbone == 'fcn_prj':
        model = FCN_prj(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool

    if args.backbone == 'fcn_expert':
            model = FCN_expert(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool
        

    if args.backbone == 'lstm':
        model = BiLSTM(args.num_classes, args.input_size)  
    
    if args.backbone == 'dilated':
        model = DilatedCNN(args.num_classes, args.input_size)  # conv1+conv2+conv3+avgpool

    if args.classifier == 'linear':
        classifier_1 = Classifier(args.classifier_input, args.num_classes)    # linear classifier
        classifier_2 = Classifier(args.classifier_input * 2, args.num_classes)    # linear classifier
        classifier = Classifier(args.classifier_input, args.num_classes)    # linear classifier

    return model, classifier_1, classifier_2, classifier

def build_loss(loss):
    if loss == 'cross_entropy':
        return cross_entropy()
    elif loss == 'reconstruction':
        return reconstruction_loss()
    elif loss == 'combined':
        return combined_loss()
    
def focal_loss(inputs, targets, num_class, alpha=0.25,  gamma=2.0):
    # 转换为 one-hot 编码
    targets_one_hot = F.one_hot(targets, num_classes=num_class).float()
    # 计算焦点损失
    inputs_softmax = F.softmax(inputs, dim=1)
    inputs_log_softmax = torch.log(inputs_softmax)
    focal_loss = -alpha * (1 - inputs_softmax) ** gamma * targets_one_hot * inputs_log_softmax

    # 适用于多分类，因此取平均
    return focal_loss.mean()

def symmetric_cross_entropy(inputs, targets,num_classes, alpha=0.1, beta=1.0):
    # 传统交叉熵损失
    ce_loss = F.cross_entropy(inputs, targets)
    # 反交叉熵损失
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    rce_loss = -torch.mean(torch.sum(F.log_softmax(inputs, dim=1) * targets_one_hot, dim=1))

    # 组合交叉熵和反交叉熵
    return alpha * ce_loss + beta * rce_loss

class combined_loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.args = args
        # self.kwargs = kwargs

    def forward(self, inputs, targets, num_class, lambda_weight=0.5, alpha=0.25, gamma=2.0, **kwargs):

        fl = focal_loss(inputs, targets, num_class, alpha, gamma)
        sce = symmetric_cross_entropy(inputs, targets, num_class)
        return lambda_weight * fl + (1 - lambda_weight) * sce

def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)  # 原地打乱索引
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def get_all_datasets(data, target):
    return k_fold(data, target)


def convert_coeff(x, eps=1e-6):
    #  计算每个复数的振幅（或模）。振幅是复数实部和虚部平方和的平方根，eps 确保不会对零取平方根。
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))

    # 计算每个复数的相位（或角度）。atan2 函数返回两个数的反正切值，这里是复数的虚部和实部，eps 确保分母不为零。
    phase = torch.atan2(x.imag, x.real + eps)
    # 将振幅和相位沿着新的最后一个维度堆叠起来。这样每个复数就被表示为一个包含其振幅和相位的向量。
    if amp.dim() == 2:
        stack_r = torch.stack((amp, phase), -1)
        stack_r = stack_r.permute(0, 2, 1)
    elif amp.dim() == 3:
        stack_r = torch.cat((amp, phase), dim=1)  # 在channels维度上合并
    return stack_r, phase

import torch.fft as fft

def evaluate_multi_experts(args, val_loader, model, classifier, classifier_1, classifier_2, loss, num_classes):

    val_loss = 0
    val_accu = 0

    sum_len = 0

    # confusion_matrix = torch.zeros(num_classes, num_classes)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    confusion_matrix_1 = torch.zeros(num_classes, num_classes)
    confusion_matrix_2 = torch.zeros(num_classes, num_classes)

    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        preds = []
        with torch.no_grad():
          
            feat_1, feat_2, val_pred = model(data)
            val_pred = classifier(val_pred)

            val_pred_1 = classifier_1(feat_1)
            val_pred_2 = classifier_2(feat_2)

            if args.loss == 'combined':
                val_loss += loss(val_pred, target, num_classes).item()
            else:
                val_loss += loss(val_pred, target).item()

            pred_1 = torch.argmax(val_pred_1.data, axis=1)

            pred_2 = torch.argmax(val_pred_2.data, axis=1)

            pred = torch.argmax(val_pred.data, axis=1)
            preds.append(pred)

            val_accu += torch.sum(pred == target)
            sum_len += len(target)

            # Update the confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # layer 1
            for t, p in zip(target.view(-1), pred_1.view(-1)):
                confusion_matrix_1[t.long(), p.long()] += 1
            # layer 2
            for t, p in zip(target.view(-1), pred_2.view(-1)):
                confusion_matrix_2[t.long(), p.long()] += 1

    # Calculate per-class accuracy
    per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
    per_class_accuracy_1 = confusion_matrix_1.diag()/confusion_matrix.sum(1)

    per_class_accuracy_2 = confusion_matrix_2.diag()/confusion_matrix.sum(1)


    # Calculate the total accuracy
    total_accuracy = val_accu / sum_len
    return val_loss / sum_len, per_class_accuracy, per_class_accuracy_1, per_class_accuracy_2, total_accuracy

def evaluate(args, val_loader, model, classifier, loss, num_classes):

    val_loss = 0
    val_accu = 0

    sum_len = 0

    # confusion_matrix = torch.zeros(num_classes, num_classes)
    confusion_matrix = torch.zeros(num_classes, num_classes)
   
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        preds = []
        with torch.no_grad():
          
            val_pred = model(data)
            val_pred = classifier(val_pred)

            if args.loss == 'combined':
                val_loss += loss(val_pred, target, num_classes).item()
            else:
                val_loss += loss(val_pred, target).item()
            
            pred = torch.argmax(val_pred.data, axis=1)
            preds.append(pred)

            val_accu += torch.sum(pred == target)
            sum_len += len(target)

            # Update the confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # Calculate per-class accuracy
    per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)

    # Calculate the total accuracy
    total_accuracy = val_accu / sum_len
    return val_loss / sum_len, per_class_accuracy, total_accuracy

def evaluate_avg(args, val_loader, model, model_feq, classifier, classifier_feq, loss, num_classes):
    val_loss = 0
    val_acc_all=0

    sum_len = 0

    confusion_matrix_all = torch.zeros(num_classes, num_classes)

    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            test_fft = fft.rfft(data, dim=-1)
            test_fft, _ = convert_coeff(test_fft)

            _, _, val_pred = model(data)
            _, _, val_pred_feq = model_feq(test_fft)
            
            val_pred = classifier(val_pred)
            val_pred_feq = classifier_feq(val_pred_feq)

            val_pred_all = (val_pred + val_pred_feq) / 2.0
            # val_pred_all = fusion_layer(val_pred, val_pred_feq)
            val_pred_all = torch.argmax(val_pred_all, axis=1)

            val_acc_all += torch.sum(val_pred_all == target)

            if args.loss == 'combined':
                val_loss += loss(val_pred, target, num_classes).item()
            else:
                val_loss += loss(val_pred, target).item()   # cross_entropy

            sum_len += len(target)

            # Update the confusion matrix

            for t, p in zip(target.view(-1), val_pred_all.view(-1)):
                confusion_matrix_all[t.long(), p.long()] += 1
            
    # Calculate per-class accuracy
    per_class_accuracy_all = confusion_matrix_all.diag()/confusion_matrix_all.sum(1)

    # Calculate the total accuracy
    avg_acc = val_acc_all / sum_len

    return val_loss / sum_len, avg_acc, per_class_accuracy_all

# 3.14 wyd 新增：
def evaluate_fusion(args, val_loader, model, model_feq, classifier, classifier_feq, fusion_layer, loss, num_classes):
    val_loss = 0
    val_acc_all=0

    sum_len = 0

    confusion_matrix_all = torch.zeros(num_classes, num_classes)

    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            test_fft = fft.rfft(data, dim=-1)
            test_fft, _ = convert_coeff(test_fft)

            _, _, val_pred = model(data)
            _, _, val_pred_feq = model_feq(test_fft)
            
            val_pred = classifier(val_pred)
            val_pred_feq = classifier_feq(val_pred_feq)

            # val_pred_all = (val_pred + val_pred_feq) / 2.0
            val_pred_all = fusion_layer(val_pred, val_pred_feq)
            val_pred_all = torch.argmax(val_pred_all, axis=1)

            val_acc_all += torch.sum(val_pred_all == target)

            if args.loss == 'combined':
                val_loss += loss(val_pred, target, num_classes).item()
            else:
                val_loss += loss(val_pred, target).item()   # cross_entropy

            sum_len += len(target)

            # Update the confusion matrix

            for t, p in zip(target.view(-1), val_pred_all.view(-1)):
                confusion_matrix_all[t.long(), p.long()] += 1
            
    # Calculate per-class accuracy
    per_class_accuracy_all = confusion_matrix_all.diag()/confusion_matrix_all.sum(1)

    # Calculate the total accuracy
    avg_acc = val_acc_all / sum_len

    return val_loss / sum_len, avg_acc, per_class_accuracy_all

def construct_graph_via_knn_cpl_nearind_gpu(data_embed, y_label, mask_label, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.95, num_real_class=2, num_iterations=1, logger=None, **kwargs):
    '''
    标签从已知的数据点“传播”到未知的数据点。
    对于未标记的样本（即那些在mask_label中标记为1的样本），    它们的标签是由它们在KNN图中的邻居的已知标签共同决定的。
    '''
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d

    # 由于两个张量在至少一个维度的大小不同（第一个和第二个维度）
    # PyTorch 会自动将这两个张量“广播”到一个共同的形状[n, n, d]
    # 在这个新形状中，每个[i, j, :]元素表示原始数据中第i个点和第j个点在特征维度上的差异
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n    # W_{ij} is the euclidean distance between samples r_i and r_j
    w = torch.exp(-w / 2)

    ## keep top-k values
    topk, indices = torch.topk(w, topk) # 保留相似度矩阵每一行最高的 topk 个相似度值，以及对应的索引
    mask = torch.zeros_like(w).to(device)   # 创建一个与 w（权重矩阵） 形状相同的全零矩阵。
    mask = mask.scatter(1, indices, 1)  # 将索引位置上的值设置为1，创建一个掩码，用于标记每行中 Top-K 的相似度值
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, knn graph 确保相似度矩阵是对称的，并将其转换为浮点数
    w = w * mask

    ## normalize
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)    # 将d_sqrt_inv沿着第 2 个维度复制n次
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)    # 将d_sqrt_inv沿着第 1 个维度复制n次
    s = d1 * w * d2 # 用归一化因子d1和d2对相似度矩阵w进行归一化

    # step2: label propagation, f = (i-\alpha s)^{-1}y
    y = torch.zeros(y_label.shape[0], num_real_class)   # 初始化一个 全0 的标签矩阵
    for i in range(len(mask_label)):    # 对于labeled数据，将对应值设置为 1，表示这些点的标签是已知的
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))    # 标签传播, torch.eye(n) 创建一个单位矩阵
    all_knn_label = torch.argmax(f, 1).cpu().numpy()    # 选择标签传播结果中概率最高的标签作为最终标签
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    # class_counter = Counter(y_label)
    class_counter = [0] * num_real_class
    # for i in range(num_real_class):
    #     class_counter[i] = 0
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i]
        else:
            class_counter[all_knn_label[i]] += 1
    
    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        # classwise_acc[i] = class_counter[i] / max(class_counter.values())
        classwise_acc[i] = class_counter[i] / max(class_counter)

    
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    # 对于准确率较高的类别，阈值会降低，而对于准确率较低的类别，阈值则相对较高。
    # 比较max_probs中的每个元素是否大于或等于这个调整后的阈值。
    # 结果是一个布尔张量cpl_mask，其元素值为True表示对应的预测概率满足条件，可以被用于生成伪标签
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))    # .ge: greater than or equal to

    return all_knn_label, end_knn_label, cpl_mask, indices


def create_logger(args, log_pkg):
    """
    :param logger_file_path:
    :return:
    """
    # 获取当前的时间
    current_time = datetime.now()
    # 将时间格式化为字符串，这里使用的格式是 年月日时分秒
    timestamp_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    
    # 创建日志文件的文件名
    # log_pkg = os.path.join(destination_folder, args.configs, args.dataset)
    if not os.path.exists(log_pkg):
        os.makedirs(log_pkg)
    log_filename = os.path.join(log_pkg, f'log_{timestamp_str}_{args.labeled_ratio}.log')

    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(log_filename)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def copy_files(files, destination_folder):
    """
    复制文件到指定文件夹中。
    
    参数:
    files (list): 要复制的文件列表。
    destination_folder (str): 目标文件夹路径。
    
    返回:
    无返回值。
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    destination_folder_with_time = os.path.join(destination_folder, current_time)
    os.makedirs(destination_folder_with_time)
    
    # 循环复制文件
    for file_path in files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_folder_with_time, file_name)
            shutil.copy(file_path, destination_path)
            print(f"Copied File: {file_path} to {destination_path}")
        else:
            print(f"warning: File {file_path} dose not exist, skipping copy.")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np


def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    feat=feat.cpu()
    x_ts = ts.fit_transform(feat.detach().numpy())

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


def plotlabels_multi_dim(S_lowDWeights, Trure_labels, name, num_class):
    # 设置散点形状
    maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 设置散点颜色
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
            'hotpink']
    feat = visual(S_lowDWeights)
    True_labels = Trure_labels.reshape((-1, 1))
    True_labels = True_labels.cpu()
    S_data = np.hstack((feat, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(num_class): 
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        # plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.scatter(X, Y, cmap='brg', s=100, alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    plt.legend()
    plt.show()

def plotlabels(S_lowDWeights, True_labels, name, num_class, name_file):
    pred_embed = S_lowDWeights.cpu()
    pred_embed = pred_embed.detach().numpy()

    True_labels = True_labels.cpu()

    label = True_labels.numpy()

    # 获取唯一的标签和它们的颜色
    unique_labels = np.unique(label)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 'tab10'是一个颜色图，适用于最多10个类别

    # 绘制每个类别的数据点
    for i, lbl in enumerate(unique_labels):
        # 选择属于当前类别的点
        indices = (label == lbl)
        subset = pred_embed[indices]

        # 绘制当前类别的点
        plt.scatter(subset[:, 0], subset[:, 1], color=colors(i), label=f'Feature{lbl}')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Featrue distribution of different categories')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.savefig(f"{name_file}.png",
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="white",
            edgecolor ='black',
            orientation ='landscape')
    # 显示图形
    # plt.show()