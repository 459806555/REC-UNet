import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1+dice_2)/2.0
        return bce + dice


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, input, target):
        smooth = 1e-5
        #input = torch.sigmoid(input)
        num = target.size(0)
        input_1 = input[:,0,:,:]
        target_1 = target[:,0,:,:]
        input_1 = input_1.view(num, -1)
        target_1 = target_1.view(num, -1)
        intersection_1 = (input_1 * target_1)
        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_1 = 1 - dice_1.sum() / num
        return  dice_1
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
def SLSIoULoss(pred_log,target,with_shape = True):

    pred = torch.sigmoid(pred_log)
    smooth = 0.0

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))




    alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

    loss = (intersection_sum + smooth) / \
           (pred_sum + target_sum - intersection_sum + smooth)
    lloss = LLoss(pred, target)


    siou_loss = alpha * loss
    if with_shape:
        loss = 1 - siou_loss.mean() + lloss
    else:
        loss = 1 - siou_loss.mean()

    return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss

class BCESLSLoss(nn.Module):
    def __init__(self):
        super(BCESLSLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)


        input_1 = input[:, 0, :, :].unsqueeze(1)
        input_2 = input[:, 1, :, :].unsqueeze(1)
        target_1 = target[:, 0, :, :].unsqueeze(1)
        target_2 = target[:, 1, :, :].unsqueeze(1)
        SLSLoss1 = SLSIoULoss(input_1,target_1)
        SLSoss2 = SLSIoULoss(input_2, target_2)
        SLSoss = ( SLSLoss1 +  SLSoss2) * 0.5
        Loss = bce +  SLSoss
        return Loss