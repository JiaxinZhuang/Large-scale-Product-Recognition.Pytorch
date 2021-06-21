# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import numpy as np

from pytorch_toolbelt.inference import tta
from collections import defaultdict
import os


def mean_class_recall(y_true, y_pred):
    """Mean class recall.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    class_recall = []
    target_uniq = np.unique(y_true)

    for label in target_uniq:
        indexes = np.nonzero(label == y_true)[0]
        recall = np.sum(y_true[indexes] == y_pred[indexes]) / len(indexes)
        class_recall.append(recall)
    return np.mean(class_recall)


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        samples = data['input']
        targets = data['label']
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    targets = []
    outputs = []
    for data in metric_logger.log_every(data_loader, 10, header):
        images = data['input']
        target = data['label']
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            img_var = torch.tensor(images, requires_grad=False).to(device=device)
            predicted = tta.tencrop_image2label(model, img_var,crop_size=(224,224))
            output = predicted
            # output = model(images)
            loss = criterion(output, target)

        acc1, acc2, acc3, acc4, acc5 = accuracy(output, target, topk=(1, 2, 3, 4, 5))
        mca = mean_class_recall(target.cpu().numpy(), torch.max(output, dim=1)[1].cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
        metric_logger.meters['acc4'].update(acc4.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['mca'].update(mca.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MCA {mca.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} Acc@3 {top3.global_avg:.3f} Acc@4 {top4.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mca=metric_logger.mca, top1=metric_logger.acc1, top2=metric_logger.acc2, top3=metric_logger.acc3, top4=metric_logger.acc4, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_submit(data_loader, model, device, args):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '=> ' + args.data_type

    K = 10
    save_root = './logs'
    correct = 0
    ct_num = 0
    counts = defaultdict(int)

    with open('./logs/submmited.csv', 'w+') as f:
        f.write('id,predict_1,confid_1,predict_2,confid_2,predict_3,confid_3,predict_4,confid_4,predict_5,confid_5,predict_6,confid_6,predict_7,confid_7,predict_8,confid_8,predict_9,confid_9,predict_10,confid_10\n')

    # for i, data in enumerate(dataloader):
        for data in metric_logger.log_every(data_loader, 10, header):
            images = data['input']
            path = data['path'][-1]
            target = data['label']
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                img_var = torch.tensor(images, requires_grad=False).to(device=device)
                predicted = tta.tencrop_image2label(model, img_var,crop_size=(224,224))
                #output = model(images)
                #predicted = output

                loss = criterion(predicted, target)
                if args.data_type == 'val':
                    acc1, acc2, acc3, acc4, acc5 = accuracy(predicted, target, topk=(1, 2, 3, 4, 5))
                    mca = mean_class_recall(target.cpu().numpy(), torch.max(predicted, dim=1)[1].cpu().numpy())

                    batch_size = images.shape[0]
                    metric_logger.update(loss=loss.item())
                    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                    metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
                    metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
                    metric_logger.meters['acc4'].update(acc4.item(), n=batch_size)
                    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                    metric_logger.meters['mca'].update(mca.item(), n=batch_size)

                predicted = torch.nn.functional.softmax(predicted, dim=1)
                confident, predicted = torch.topk(predicted, k=K, dim=1, largest=True, sorted=True)

                #output = model(images)


            for idx, img_path in enumerate(path):
                fname = os.path.basename(img_path)
                p = predicted[idx]
                c = confident[idx]
                f.write(fname)
                for i in range(K):
                    f.write(","+str(p[i].item())+ "," +"%.4f"%c[i].item())
                f.write("\n")

    if args.data_type == 'val':
        metric_logger.synchronize_between_processes()
        print('* MCA {mca.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} Acc@3 {top3.global_avg:.3f} Acc@4 {top4.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mca=metric_logger.mca, top1=metric_logger.acc1, top2=metric_logger.acc2, top3=metric_logger.acc3, top4=metric_logger.acc4, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
