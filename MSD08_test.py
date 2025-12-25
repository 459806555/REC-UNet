# -*- coding: utf-8 -*-

from PIL import Image
import time
import os

import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


import joblib


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from MSD08_dataset import Dataset

from MSD08_utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score,ppv,iou_score_batch,sensitivity,p_value_test,precision,F1_score,accuracy
import MSD08_utilities.losses as losses
from MSD08_utilities.utils import str2bool, count_params
import pandas as pd
import REC_UNet

arch_names = list(REC_UNet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='models name: (default: arch+timestamp)')
    parser.add_argument('--deepsupervision', default=None,
                        help='models name: (default: arch+timestamp)')
   
    parser.add_argument('--arch', '-a', metavar='ARCH', default='REC_UNet',
                        choices=arch_names,
                        help='models architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    
    parser.add_argument('--dataset', default="MSD",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 30)')

    
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_test(output, target):

    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
        output = output > 0.5
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:, 0, :, :]
    input_2 = output[:, 1, :, :]

    target_1 = target[:, 0, :, :]
    target_2 = target[:, 1, :, :]

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (input_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (input_2.sum() + target_2.sum() + smooth)

    return dice_1, dice_2
def test(args, val_loader, model, criterion):
    # Metrics with suffix "1" = hepatic vessel segmentation; suffix "2" = hepatic tumor segmentation. Only focus on tumor segmentation here.
    losses = AverageMeter()
    ious_1 = AverageMeter()
    ious_2 = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    recall_1s = AverageMeter()
    recall_2s = AverageMeter()
    accuracy_1s = AverageMeter()
    accuracy_2s = AverageMeter()
    precision_1s = AverageMeter()
    precision_2s = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou_1 = iou_score_batch(output[:, 0, :, :], target[:, 0, :, :])
                iou_2 = iou_score_batch(output[:, 1, :, :], target[:, 1, :, :])
                dice_1 = dice_test(output, target)[0]
                dice_2 = dice_test(output, target)[1]
                recall_1 = sensitivity(output[:, 0, :, :], target[:, 0, :, :])
                recall_2 = sensitivity(output[:, 1, :, :], target[:, 1, :, :])
                accuracy_1 = accuracy(output[:, 0, :, :], target[:, 0, :, :])
                accuracy_2 = accuracy(output[:, 1, :, :], target[:, 1, :, :])
                precision_1 = precision(output[:, 0, :, :], target[:, 0, :, :])
                precision_2 = precision(output[:, 1, :, :], target[:, 1, :, :])
            losses.update(loss.item(), input.size(0))
            ious_1.update(iou_1, input.size(0))
            ious_2.update(iou_2, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))
            recall_1s.update(torch.tensor(recall_1), input.size(0))
            recall_2s.update(torch.tensor(recall_2), input.size(0))
            accuracy_1s.update(torch.tensor(accuracy_1), input.size(0))
            accuracy_2s.update(torch.tensor(accuracy_2), input.size(0))
            precision_1s.update(torch.tensor(precision_1), input.size(0))
            precision_2s.update(torch.tensor(precision_2), input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou_1', ious_1.avg),
        ('iou_2', ious_2.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg),
        ('recall_1', recall_1s.avg),
        ('recall_2', recall_2s.avg),
        ('accuracy_1', accuracy_1s.avg),
        ('accuracy_2', accuracy_2s.avg),
        ('precision_1', precision_1s.avg),
        ('precision_2', precision_2s.avg),
    ])
    return log
def main():
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')




    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True
    val_img_paths = glob('./MSD_data/test_image/*')
    val_mask_paths = glob('./MSD_data/test_label/*')
    print("val_num:%s" % str(len(val_img_paths)))

    # create models
    print("=> creating models %s" % args.arch)
    model = REC_UNet.REC_UNet(args)
    model = torch.nn.DataParallel(model).cuda()
    #For loading the model weights from the final training epochs
    path = r""
    model.load_state_dict(torch.load(path))

    print(count_params(model))


    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    with torch.no_grad():
        val_log = test(args, val_loader, model, criterion)
# Metrics with suffix "1" = hepatic vessel segmentation; suffix "2" = hepatic tumor segmentation. Only focus on tumor segmentation here.
        print(
            'loss %.4f - dice %.4f - iou %.4f - recall %.4f  - acc %.4f - precision %.4f'
            % (val_log['loss'], val_log['dice_2'], val_log['iou_2'],
               val_log['recall_2'], val_log['accuracy_2'], val_log['precision_2']))


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

main()




