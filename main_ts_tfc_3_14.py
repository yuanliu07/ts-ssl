import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import queue
import time
from datetime import datetime
from collections import Counter
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ts_data.dataset_load import *
from ts_data.data_load import normalize_per_series, fill_nan_value
from ts_model.loss_3_9 import sup_contrastive_loss
from ts_model.loss import KL
from ts_model.model import ProjectionHead, FusionLayer
from ts_utils import build_experts_in_on_model, set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate_multi_experts, convert_coeff, create_logger, copy_files, plotlabels, label_propagation, evaluate_fusion       # 3.14 wyd

np.random.seed(123)  # 为NumPy设置随机种子
torch.manual_seed(123)  # 为PyTorch设置随机种子

# 如果使用的是 CUDA，则还需要以下代码来确保CUDA的行为也是确定性的
torch.cuda.manual_seed(123)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='./results/', type=str, help='log file path to save result')

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn_expert', help='encoder backbone, fcn; tc, dilated, lstm, fcn_prj, fcn_expert')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
    parser.add_argument('--log_epoch', type=int, default=100, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='FDB',
                        help='dataset, imbalanced: PLAID(in ucr), sleepEDF, SemgHandGenderCh2  ECG, EMG; balanced: GestureMidAirD1(in ucr). FD-A, FD-B, Gesture, Epilepsy, SleepEEG, GunPointOldVersusYoung(UCR)') 
    parser.add_argument('--configs', type=str, default='FDB', help='config file: UCR; SleepEDF')

    parser.add_argument('--dataroot', type=str, default='C:/Users/YUAN/Desktop/TS-SSL/ts_data/dataset', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=300, help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--stage_two_epoch', type=int, default=600, help='the stage that model starts to learn from the pseudo label from each other')
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3, 5 for ECG')
    parser.add_argument('--knn_num_tem', type=int, default=40, help='10, 20, 50')
    parser.add_argument('--knn_num_feq', type=int, default=30, help='10, 20, 50')

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.3, help='weight for supervised contrastive loss: 0.05 or 0.005')
    parser.add_argument('--sup_con_lambda', type=float, default=0.05, help='weight for pseudo contrastive loss: 0.05 or 0.005')
    parser.add_argument('--mlp_head', type=bool, default=True, help='head project')
    parser.add_argument('--temperature', type=float, default=50, help='20, 50')

    # training setup
    parser.add_argument('--loss', type=str, default='combined', help='loss function, cross_entropy, combined')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()
    
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    log_epoch = args.log_epoch

    exec(f'from config_files.{args.configs}_Configs import Config as Configs')  
    configs = Configs()

    # KL 
    factor_1 = configs.factor_1
    factor_2 = configs.factor_2
    factor_3 = configs.factor_3
           
    # args.loss = configs.loss.loss_type
    set_seed(args)

    # save the current running files
    files_to_copy = [__file__, "ts_utils.py", "ts_model/model_o.py", "ts_model/loss_3_9.py"]  # 要复制的文件列表
    destination_folder = os.path.join("saved_files_results", args.configs, args.dataset)
    
    copy_files(files_to_copy, destination_folder)
    logger = create_logger(args, destination_folder)

    logger.info('-' * 50)
    logger.info(__file__)
    # 将所有参数及其实际使用的值记录到logger中
    for arg in vars(args):
        logger.info(f"Argument {arg}: {getattr(args, arg)}")

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list = build_dataset(args, logger)  
    args.num_classes = num_classes
    logger.info(f'number of classes: {num_classes}\t class distribution: {num_class_list}')
    logger.info('-' * 50)

    model, classifier_1, classifier_2, classifier = build_experts_in_on_model(args)
    projection_head = ProjectionHead(input_dim=128)

    model, classifier = model.to(device), classifier.to(device)
    classifier_1, classifier_2 = classifier_1.to(device), classifier_2.to(device)
    projection_head = projection_head.to(device)

    loss = build_loss(args.loss).to(device)

    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    projection_head_init_state = projection_head.state_dict()

    # feq
    args.input_size = 2

    model_feq, classifier_feq_1, classifier_feq_2, classifier_feq = build_experts_in_on_model(args)
    projection_head_feq = ProjectionHead(input_dim=128)

    fusion_layer = FusionLayer().to(device)      # 3.14 wyd

    model_feq, classifier_feq = model_feq.to(device), classifier_feq.to(device)

    classifier_feq_1, classifie_feq_2 = classifier_feq_1.to(device), classifier_feq_2.to(device)

    projection_head_feq = projection_head_feq.to(device)

    loss_feq = build_loss(args.loss).to(device)

    model_feq_init_state = model_feq.state_dict()
    classifier_feq_init_state = classifier_feq.state_dict()
    projection_head_feq_init_state = projection_head_feq.state_dict()

    is_projection_head = args.mlp_head


    optimizer = torch.optim.Adam(
                                list(model.parameters()) + list(classifier.parameters()) + list(projection_head.parameters()) + list(fusion_layer.parameters()) 
                                + list(model_feq.parameters()) + list(classifier_feq.parameters()) + list(projection_head_feq.parameters()) , 
                                lr=args.lr # 学习率
                            )        # 3.14 wyd 把optimizer_feq 删掉了。后面对应的关于optimizer_feq的操作也删除
    
    # print('start ssl on {}'.format(args.dataset))
    logger.info('start ssl on {}'.format(args.dataset))

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    test_acc_avg_k_folds = []

    # i: fold
    for i, train_dataset in enumerate(train_datasets):

        t = time.time()

        model.load_state_dict(model_init_state)
        classifier.load_state_dict(classifier_init_state)
        projection_head.load_state_dict(projection_head_init_state)

        logger.info('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

        if args.labeled_ratio == 1:
            train_all_split = train_dataset
            y_label_split = train_target

        else:
            train_labeled, train_unlabeled, y_labeled, y_unlabeled = train_test_split(train_dataset, train_target,
                                                                                    test_size=(1 - args.labeled_ratio),
                                                                                    random_state=args.random_seed)

            mask_labeled = np.zeros(len(y_labeled)) # [0, 0, ...., 0]
            mask_unlabeled = np.ones(len(y_unlabeled))  # [1, 1, ...., 1]
            mask_train = np.concatenate([mask_labeled, mask_unlabeled]) # [0, 0, ...., 1]

            train_all_split = np.concatenate([train_labeled, train_unlabeled])
            y_label_split = np.concatenate([y_labeled, y_unlabeled])

        x_train_all, y_train_all = shuffler(train_all_split, y_label_split)
        mask_train, _ = shuffler(mask_train, mask_train)        
        y_train_all[mask_train == 1] = -1  ## Generate unlabeled data   将所有unlabeled数据类别标注为-1

        # 使用 Counter 统计每个类别出现的次数
        class_counts = Counter(y_train_all)
        # 将结果存入一个列表中
        class_counts_list = [class_counts[i] for i in range(len(class_counts))]
        logger.info(f'Initial distribution of labeled y :{class_counts_list}')

        train_fft = fft.rfft(torch.from_numpy(x_train_all), dim=-1)
        train_fft, _ = convert_coeff(train_fft)
        train_fft = train_fft.to(device)
        x_train_labeled_all_feq = train_fft[mask_train == 0]

        x_train_all = torch.from_numpy(x_train_all).to(device)
        y_train_all = torch.from_numpy(y_train_all).to(device).to(torch.int64)

        if x_train_all.dim() == 3:
            x_train_labeled_all = x_train_all[mask_train == 0] # HAR
        elif x_train_all.dim() == 2:
            x_train_labeled_all = torch.unsqueeze(x_train_all[mask_train == 0], 1)  # 增加labeled数据的维度

        y_train_labeled_all = y_train_all[mask_train == 0]

        train_set_labled = Load_Dataset(x_train_labeled_all, y_train_labeled_all, configs)
        train_set = Load_Dataset(x_train_all, y_train_all, configs)    # x_train_all 包含 labeled 和 unlabeled data
        val_set = Load_Dataset(torch.from_numpy(val_dataset).to(device), torch.from_numpy(val_target).to(device).to(torch.int64), configs)
        test_set = Load_Dataset(torch.from_numpy(test_dataset).to(device), torch.from_numpy(test_target).to(device).to(torch.int64), configs)
        
        batch_size_labeled = 128
        while x_train_labeled_all.shape[0] < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if x_train_labeled_all.shape[0] < 16:
            batch_size_labeled = 16

        train_labeled_loader = DataLoader(train_set_labled, batch_size=batch_size_labeled, num_workers=0,
                                          drop_last=False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        val_fft = fft.rfft(torch.from_numpy(val_dataset), dim=-1)
        val_fft, _ = convert_coeff(val_fft)
        val_set_feq = Load_Dataset(val_fft.to(device),torch.from_numpy(val_target).to(device).to(torch.int64), configs)

        test_fft = fft.rfft(torch.from_numpy(test_dataset), dim=-1)
        test_fft, _ = convert_coeff(test_fft)
        test_set_feq = Load_Dataset(test_fft.to(device), torch.from_numpy(test_target).to(device).to(torch.int64), configs)

        val_loader_feq = DataLoader(val_set_feq, batch_size=args.batch_size, num_workers=0)
        test_loader_feq = DataLoader(test_set_feq, batch_size=args.batch_size, num_workers=0)

        train_loss = []
        train_accuracy = []
        train_loss_feq = []
        train_accuracy_feq = []

        num_steps = args.epoch // args.batch_size

        last_loss_tem = float('inf')
        last_loss_feq = float('inf')

        stop_count = 0
        increase_count = 0

        num_steps = train_set.__len__() // args.batch_size
        if num_steps == 0:
            num_steps = num_steps + 1

        min_val_loss_tem = float('inf')
        min_val_loss_tem_stage_2 = float('inf')
        val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0
        min_val_loss_feq = float('inf')
        min_val_loss_feq_stage_2 = float('inf')

        test_accuracy_tem = 0
        end_val_epoch_tem = 0
        test_accuracy_feq = 0
        end_val_epoch_feq = 0

        tem_better_count = 0
        feq_better_count = 0

        queue_train_x = queue.Queue(args.queue_maxsize)
        queue_train_y = queue.Queue(args.queue_maxsize)
        queue_train_mask = queue.Queue(args.queue_maxsize)

        queue_train_x_feq = queue.Queue(args.queue_maxsize)
        queue_train_y_feq = queue.Queue(args.queue_maxsize)

        for epoch in range(args.epoch):

            if stop_count == 80 or increase_count == 80:
                logger.info('model convergent at epoch {}, early stopping'.format(epoch))
                break

            epoch_train_loss = 0
            epoch_train_acc = 0
            num_iterations = 0

            model.train()
            classifier.train()
            projection_head.train()
            model_feq.train()
            classifier_feq.train()
            projection_head_feq.train()

            fusion_layer.train()     # 3.14 wyd

            if epoch <= args.warmup_epochs:
            # if False:
                
                for ind, (x, y) in enumerate(train_labeled_loader):
                    if x.shape[0] < 2:
                        continue
                    if (num_iterations + 1) * batch_size_labeled < x_train_labeled_all_feq.shape[0]:
                        x_feq = x_train_labeled_all_feq[
                                num_iterations * batch_size_labeled: (num_iterations + 1) * batch_size_labeled]
                    else:
                        x_feq = x_train_labeled_all_feq[num_iterations * batch_size_labeled:]

                    optimizer.zero_grad()

                    # feat_1: 第 1 层conv的特征
                    # feat_2: 第 2 层conv的特征
                    # pred_embed: 第 3 层conv的特征，也即模型的最后一层
                    # 将每层的特征提取出来，分别输入 classifier，得到对应层的 logits，
                    # 从而实现两个模型相同层（即：时域模型的conv1和频域模型的conv1、时域模型的conv2与频域模型的conv2，...）之间的对齐

                    feat_1, feat_2, pred_embed = model(x)
                    feat_1_feq, feat_2_feq, pred_embed_feq = model_feq(x_feq)

                    # 引出一个单独的分支，用于计算对比损失
                    preject_head_embed = projection_head(pred_embed)
                    preject_head_embed_feq = projection_head_feq(pred_embed_feq)

                    pred = classifier(pred_embed)
                    pred_feq = classifier_feq(pred_embed_feq)

                    pred_layer1 = classifier_1(feat_1)
                    pred_layer2 = classifier_2(feat_2)

                    pred_feq_layer1 = classifier_feq_1(feat_1_feq)
                    pred_feq_layer2 = classifier_feq_2(feat_2_feq)

                    ######################
                    final_output = fusion_layer(pred, pred_feq)     # 3.14 wyd
                    ######################
                    if args.loss == 'combined':
                        # step_loss = loss(pred, y, args.num_classes)
                        # step_loss_feq = loss_feq(pred_feq, y, args.num_classes)
                        step_loss_final = loss(final_output, y, args.num_classes)
                    else:
                        # step_loss = loss(pred, y)
                        # step_loss_feq = loss_feq(pred_feq, y)
                        step_loss_final = loss(final_output, y)

                    ######################
                    # 新增：时域与频域模型对应的卷积层之间，计算KL损失，对其二者的logits，
                    loss_cross_learn = KL([pred_layer1, pred_feq_layer1], factor_1) + KL([pred_layer2, pred_feq_layer2], factor_2) + KL([pred, pred_feq], factor_3)
                    ######################
                   
                    if len(y) > 1:
                        batch_sup_contrastive_loss = sup_contrastive_loss(
                            embd_batch=preject_head_embed,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=20)

                        # step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                        batch_sup_contrastive_loss_feq = sup_contrastive_loss(
                            embd_batch=preject_head_embed_feq,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=20)

                        # step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_mu

                        step_loss_final = step_loss_final + batch_sup_contrastive_loss_feq * args.sup_con_mu + batch_sup_contrastive_loss * args.sup_con_mu

                    all_loss = step_loss_final + loss_cross_learn

                    # all_loss = step_loss_feq + step_loss + loss_cross_learn

                    all_loss.backward()

                    optimizer.step()

                    num_iterations = num_iterations + 1
            else:
                for ind, (x, y) in enumerate(train_loader):
                    if x.shape[0] < 2:
                        continue
                    if (num_iterations + 1) * args.batch_size < train_set.__len__():
                        x_feq = train_fft[
                                num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                        y_feq = y_train_all[
                                num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                        mask_train_batch = mask_train[
                                           num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]    # 提取一个batch的mask
                    else:
                        x_feq = train_fft[num_iterations * args.batch_size:]
                        y_feq = y_train_all[num_iterations * args.batch_size:]
                        mask_train_batch = mask_train[num_iterations * args.batch_size:]

                    optimizer.zero_grad()

                    feat_1, feat_2, pred_embed = model(x)
                    feat_1_feq, feat_2_feq, pred_embed_feq = model_feq(x_feq)

                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)
                        preject_head_embed_feq = projection_head_feq(pred_embed_feq)

                    mask_cpl_batch = torch.tensor([False] * len(mask_train_batch)).to(device)
                    mask_cpl_batch_feq = torch.tensor([False] * len(mask_train_batch)).to(device)

                    if epoch > args.warmup_epochs:
                    # if True:
                        if not queue_train_x.full():
                            queue_train_x.put(preject_head_embed.detach())
                            queue_train_y.put(y)
                            queue_train_mask.put(mask_train_batch)

                            queue_train_x_feq.put(preject_head_embed_feq.detach())
                            queue_train_y_feq.put(y_feq)

                        if queue_train_x.full():    
                            # 当队列 queue_train_x 满时，从队列中提取所有数据、标签和掩码。
                            # 使用 torch.cat 和 np.concatenate 将提取的数据、标签和掩码组合成单个张量或数组。
                            train_x_allq = queue_train_x.queue
                            train_y_allq = queue_train_y.queue
                            train_mask_allq = queue_train_mask.queue

                            train_x_allq_feq = queue_train_x_feq.queue
                            train_y_allq_feq = queue_train_y_feq.queue
                            
                            # 构建KNN图 ts_utils.py
                            # end_knn_label 和 mask_cpl_knn 分别是基于KNN图更新后的标签和掩码。
                            _, end_knn_label, mask_cpl_knn, _ = label_propagation(
                                train_x_allq, train_y_allq,
                                train_mask_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_tem, num_iterations=num_iterations, logger=logger)
                            
                            knn_result_label = torch.tensor(end_knn_label).to(device).to(torch.int64)

                            y[mask_train_batch == 1] = knn_result_label[(len(knn_result_label) - len(y)):][
                                mask_train_batch == 1]  # 根据 end_knn_label 更新 unlabeled data 的标签
                            
                            mask_cpl_batch[mask_train_batch == 1] = mask_cpl_knn[(len(mask_cpl_knn) - len(y)):][
                                mask_train_batch == 1]

                            _, end_knn_label_feq, mask_cpl_knn_feq, _ = label_propagation(
                                train_x_allq_feq, train_y_allq_feq,
                                train_mask_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_feq, num_iterations=num_iterations, logger=logger)
                            
                            knn_result_label_feq = torch.tensor(end_knn_label_feq).to(device).to(torch.int64)

                            y_feq[mask_train_batch == 1] = knn_result_label_feq[(len(knn_result_label_feq) - len(y)):][
                                mask_train_batch == 1]
                            
                            mask_cpl_batch_feq[mask_train_batch == 1] = \
                                mask_cpl_knn_feq[(len(knn_result_label_feq) - len(y)):][mask_train_batch == 1]

                            _ = queue_train_x.get()
                            _ = queue_train_y.get()
                            _ = queue_train_mask.get()

                            _ = queue_train_x_feq.get()
                            _ = queue_train_y_feq.get()


                    mask_clean = [True if mask_train_batch[m] == 0 else False for m in range(len(mask_train_batch))]
                    mask_select_loss = [False for m in range(len(y))]

                    mask_true_and_unpropogated = [False for m in range(len(y))]
                    
                    for m in range(len(mask_train_batch)):
                        if mask_train_batch[m] == 0:
                            mask_select_loss[m] = True  # true label
                          
                        else:
                            if mask_cpl_batch[m]:
                                mask_select_loss[m] = True
                           
                    pred = classifier(pred_embed)  # [batch_size, num_class]
                    pred_feq = classifier_feq(pred_embed_feq)

                    pred_layer1 = classifier_1(feat_1)
                    pred_layer2 = classifier_2(feat_2)

                    pred_feq_layer1 = classifier_feq_1(feat_1_feq)
                    pred_feq_layer2 = classifier_feq_2(feat_2_feq)
                    ######################
                    final_output = fusion_layer(pred, pred_feq) # 3.14 wyd
                    ######################
                    # 新增：时域与频域模型对应的卷积层之间，计算KL损失，对其二者的logits，
                    loss_cross_learn = KL([pred_layer1, pred_feq_layer1], 0.5) + KL([pred_layer2, pred_feq_layer2], 0.5)  + KL([pred, pred_feq], 0.5)
                    #####################
                    mask_unpropagated = y != -1 # true label + pseudo label
                    mask_unpropagated_feq = y_feq != -1
                           
                    mask_true_and_unpropogated = y[mask_select_loss] != -1  # pseudo label for supervised contrastive loss
                    mask_true_and_unpropogated = [a and b for a, b in zip(mask_select_loss, mask_unpropagated)]

                    if args.loss == 'combined':     # 3.14 wyd
                        step_loss =  loss(pred[mask_true_and_unpropogated], y_feq[mask_true_and_unpropogated], args.num_classes) * 0.5
                        step_loss_feq = loss_feq(pred_feq[mask_true_and_unpropogated], y[mask_true_and_unpropogated], args.num_classes) * 0.5
                        
                        step_loss_fusion = loss(final_output[mask_select_loss], y[mask_select_loss], args.num_classes)

                        # step_loss = loss(pred[mask_select_loss], y_feq[mask_select_loss], args.num_classes) + loss(pred[mask_unpropagated], y_feq[mask_unpropagated], args.num_classes) * 0.3
                        # step_loss_feq = loss_feq(pred_feq[mask_select_loss], y[mask_select_loss], args.num_classes) + loss_feq(pred_feq[mask_unpropagated], y[mask_unpropagated], args.num_classes) * 0.3

                    else:     # 3.14 wyd
                        step_loss =  loss(pred[mask_true_and_unpropogated], y_feq[mask_true_and_unpropogated]) * 0.5
                        step_loss_feq = loss_feq(pred_feq[mask_true_and_unpropogated], y[mask_true_and_unpropogated]) * 0.5

                        step_loss_fusion = loss(final_output[mask_select_loss], y_feq[mask_select_loss])

                        # step_loss = loss(pred[mask_select_loss], y_feq[mask_select_loss]) + loss(pred[mask_unpropagated], y_feq[mask_unpropagated]) * 0.3
                        # step_loss_feq = loss_feq(pred_feq[mask_select_loss], y[mask_select_loss]) + loss_feq(pred_feq[mask_unpropagated], y[mask_unpropagated]) * 0.3

                    if epoch > args.warmup_epochs:
                    # # if True:
                        if len(y[mask_train_batch == 0]) > 1:
                            batch_sup_contrastive_loss = sup_contrastive_loss(
                                embd_batch=preject_head_embed[mask_train_batch == 0],
                                labels=y[mask_train_batch == 0],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)

                            batch_sup_contrastive_loss_feq = sup_contrastive_loss(
                                embd_batch=preject_head_embed_feq[mask_train_batch == 0],
                                labels=y_feq[mask_train_batch == 0],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)
                            
                            step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                            step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_mu 
                        
                        # 通过[mask_true_and_unpropogated]，将 “有原始标注的样本”和“未生成伪标签的样本”掩盖掉
                        # 只取其中：伪标签对应的样本（preject_head_embed是这些样本的特征），然后计算 时域 和 频域 的对比损失
                        # 因为伪标签并不一定可靠，所以给这部分损失设置一个较小的权重：args.sup_con_lambda 
                            
                        if len(y[mask_true_and_unpropogated]) > 1:
                            batch_sup_contrastive_loss_pseudo = sup_contrastive_loss(
                                embd_batch=preject_head_embed[mask_true_and_unpropogated],
                                labels=y[mask_true_and_unpropogated],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)

                            step_loss = step_loss + batch_sup_contrastive_loss_pseudo * args.sup_con_lambda

                            batch_sup_contrastive_loss_feq_pseudo = sup_contrastive_loss(
                                embd_batch=preject_head_embed_feq[mask_true_and_unpropogated],
                                labels=y_feq[mask_true_and_unpropogated],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)
                            
                            step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq_pseudo * args.sup_con_lambda

                    all_loss = step_loss_feq + step_loss + step_loss_fusion     # 3.14 wyd
                    # all_loss = step_loss_feq + step_loss

                    all_loss.backward()

                    optimizer.step()

                    num_iterations += 1

                # if epoch % 100 == 0:
                #     y_with_pseudo[mask_train == 1] = y[mask_train == 1]
                #     train_set.update_labels(y)
                #     # 更新DataLoader
                #     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                    
            ##################### Validation ########################
            
            model.eval()
            classifier.eval()
            projection_head.eval()
            model_feq.eval()
            classifier_feq.eval()
            projection_head_feq.eval()

            fusion_layer.eval()     # 3.14 wyd

            val_loss_tem, per_class_accuracy, per_class_accuracy_1, per_class_accuracy_2, val_accu_tem = evaluate_multi_experts(args, val_loader, model, classifier, classifier_1, classifier_2, loss, args.num_classes)

            val_loss_feq, per_class_accuracy_feq, per_class_accuracy_1_feq, per_class_accuracy_2_feq, val_accu_feq = evaluate_multi_experts(args, val_loader_feq, model_feq, classifier_feq, classifier_feq_1, classifier_feq_2, loss_feq,  args.num_classes)

            val_loss_avg, val_acc_avg, val_per_class_accuracy_avg = evaluate_fusion(args, val_loader, model, model_feq, classifier, classifier_feq, fusion_layer, loss, num_classes)     # 3.14 wyd

            # 统计时域和频域模型表现更优的次数
            if val_accu_tem >= val_accu_feq: 
                tem_better_count += 1
            else:
                feq_better_count += 1

            if epoch > args.warmup_epochs:
                # 如果模型的loss在一定epoch（实验中设置为80）内，损失没有下降，则停止训练
                if (abs(last_loss_tem - val_loss_tem) <= 1e-4) and (abs(last_loss_feq - val_loss_feq) <= 1e-4):
                    stop_count += 1
                else:
                    stop_count = 0
            
                if (val_loss_feq > last_loss_feq) or (val_loss_tem > last_loss_tem):
                    increase_count += 1
                else:
                    increase_count = 0

            last_loss_tem = val_loss_tem
            last_loss_feq = val_loss_feq

            if epoch % args.log_epoch == 0:

                logger.info("epoch : {} \t valid_accuracy_tem : {:.5f} \t valid_accuracy_feq : {:.5f} \t valid_accuracy_all : {:.5f}".format(epoch, val_accu_tem, val_accu_feq, val_acc_avg))

        t = time.time() - t
        train_time += t

        ##################### Test ########################
        # 分别测试 时域 和 频域 两个子模型
        '''
        test_per_class_accuracy_1_tem: 将第一个 conv 层的特征输入 classifier 1，得到对于每一类的分类结果（准确率）
        test_per_class_accuracy_2_tem: 将第二个 conv 层的特征输入 classifier 2，得到对于每一类的分类结果（准确率）
        test_accuracy_tem：最后一层 conv 特征输入classifier，即 整个时域模型，得到对于每一类的分类结果（准确率）
        feq 同理
        '''
        test_loss_tem, test_per_class_accuracy_tem, test_per_class_accuracy_1_tem, test_per_class_accuracy_2_tem, test_accuracy_tem = evaluate_multi_experts(args, test_loader, model, classifier, classifier_1, classifier_2, loss,  args.num_classes)
        
        test_loss_feq, test_per_class_accuracy_feq, test_per_class_accuracy_1_feq, test_per_class_accuracy_2_feq, test_accuracy_feq = evaluate_multi_experts(args, test_loader_feq, model_feq, classifier_feq, classifier_feq_1, classifier_feq_2,
                                                                loss_feq,  args.num_classes)
        
        # 综合评估整个模型
        test_loss_avg, test_acc_avg, test_per_class_accuracy_avg = evaluate_fusion(args, test_loader, model, model_feq, classifier, classifier_feq, fusion_layer, loss, num_classes)         # 3.14 wyd
        
        # 保存当前（第k折）的测试结果
        test_acc_avg_k_folds.append(test_acc_avg)

        logger.info(f'Test accuracy of {i}-th fold training: {test_acc_avg}')

    # k-折结果的平均值
    test_acc_avg_k_folds = torch.Tensor(test_acc_avg_k_folds)
    
    mean_acc = torch.mean(test_acc_avg_k_folds).item()

    logger.info(f"Traning Done: time (seconds) = {round(train_time, 3)}")

    # 输出：最后的测试结果，以及模型训练时间
    logger.info(f"Test accuracy = {mean_acc}, traning time (seconds) = {round(train_time, 3)}")

    # 模型对于每个类别的分类准确率
    logger.info(f'Test accuracy of per class: {test_per_class_accuracy_avg}')


    logger.info(f'tem better times: {tem_better_count}\t feq better times: {feq_better_count}')
   
    logger.info(f'Time \t Per-class Layer 1: \t {test_per_class_accuracy_1_tem}')
    logger.info(f'Time \t Per-class Layer 2: \t {test_per_class_accuracy_2_tem}')
    logger.info(f'Time \t Per-class Layer 3: \t {test_per_class_accuracy_tem}')

    logger.info(f'Feq \t Per-class Layer 1: \t {test_per_class_accuracy_1_feq}')
    logger.info(f'Feq \t Per-class Layer 2: \t {test_per_class_accuracy_2_feq}')
    logger.info(f'Feq \t Per-class Layer 3: \t {test_per_class_accuracy_feq}')

    try:
        # 定义 CSV 文件名
        output_file = "model_accuracy.csv"
        
        # 检查文件是否存在，不存在则添加标题行
        file_exists = os.path.isfile(output_file)
        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ['Dataset', 'Accuracy', 'Distribution']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # 文件不存在，写入头

            writer.writerow({'Dataset': args.dataset, 'Accuracy': mean_acc, 'Distribution': num_class_list})
    except:
        print('wrong while saving results')

    # visualize

    # 图例名称
    Label_Com = ['a', 'b', 'c', 'd']
    # 设置字体格式
    font1 = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 32,
            }
    
    # plotlabels(visual(pred_embed), y, '(a)', args.num_classes)
    # plotlabels(visual(pred_embed_feq), y, '(b)', args.num_classes)
    
    plotlabels(pred_embed, y, '(a)', args.num_classes, os.path.join(destination_folder, 'time'))
    plotlabels(pred_embed_feq, y, '(b)', args.num_classes, os.path.join(destination_folder, 'feq'))
