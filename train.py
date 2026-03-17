import os
import argparse
from datetime import datetime
import numpy as np
import random
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = '2,4'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.NESS_Net import NESS_Net

from data import get_loader

from utils import clip_gradient, adjust_lr, label_edge_prediction, visualize_prediction
from losses import smooth_loss, ssim_loss
import torch.backends.cudnn as cudnn
import warnings


warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)   # PyTorch CPU
torch.cuda.manual_seed(SEED)          # GPU
torch.cuda.manual_seed_all(SEED)      
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False     

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=35, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')

parser.add_argument('--sm_loss_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--edge_loss_weight', type=float, default=0.1, help='weight for edge loss')
parser.add_argument('--ssim_loss_weight', type=float, default=0.85, help='ssim loss weight')

parser.add_argument('--general_stage', action='store_true', help='general stage without consistency training')
parser.add_argument('--keep_teacher', action='store_true', help='keep teacher model during training')
parser.add_argument('--mom_coef', type=float, default=0.999, help='momentum coefficient for teacher model updating')

parser.add_argument('--data_dir', default='./dataset', help='path where to save training data')
parser.add_argument('--output_dir', default='./checkpoints', help='path where to save trained models')
parser.add_argument('--vis_dir', default='./vis', help='path where to save visualizations, empty for no saving')
parser.add_argument('--general_model', default='./checkpoints/NESS_general_30.pth', help='path where to save warm-up model')
parser.add_argument('--gpu_id', default='all', help='gpus used to train')
parser.add_argument('--load', type=str, default='swin_base_patch4_window12_384_22k.pth', help='pth used to train')

opt = parser.parse_args()


CE = torch.nn.BCELoss()
smooth_loss = smooth_loss.smoothness_loss(size_average=True)
SSIM = ssim_loss.SSIM()
device = torch.device("cuda:0")

def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts

    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    # predict = torch.sigmoid(predict_)
    smooth = 1
    p =2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0],-1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2+smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num/den

    return loss.mean()


@torch.no_grad()
def teacher_model_update(teacher_model, student_model, m=0.999):
    student_model_dict = student_model.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = student_model_dict[key] * (1 - m) + value * m
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)
    return teacher_model

def iou_loss(pred, target):
    intersection = (pred * target).float().sum((2, 3))  
    union = (pred + target).float().sum((2, 3))/2
    iou = (intersection + 1e-16) / (union + 1e-16)
    iou_loss = 1 - iou.mean()
    return iou_loss

class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)#if soft
        return iou_loss(pred, target)
    
def train(train_loader, model, teacher_model, optimizer, epoch):
    model.train()
    IOULoss = IOU()
    torch.autograd.set_detect_anomaly(True)
    if not opt.general_stage:
        teacher_model.train()

    total_step = len(train_loader)
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        if opt.general_stage:
            images, depths, gts, masks, grays, image_edges, depth_edges = pack
        else:
            images, depths, gts, masks, grays, image_edges, depth_edges, \
            images_t, depths_t, image_edges_t, depth_edges_t, flip_flag = pack

        images = Variable(images)
        gts = Variable(gts)
        masks = Variable(masks)
        grays = Variable(grays)
        depths = Variable(depths)   
        
        images = images.to(device)
        depths = depths.to(device)
        gts = gts.to(device)
        masks = masks.to(device)
        grays = grays.to(device)
        image_edges = image_edges.to(device)
        depth_edges = depth_edges.to(device)

        sal1, edge_map, sal2, edge_map_depth, img_edges, dep_edges = model(images, depths, image_edges, depth_edges)
        
        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        
        
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob = sal2_prob * masks
        
        smoothLoss_cur1 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal1), grays)
        
        if opt.general_stage:
            sal_loss1 = ratio * CE(sal1_prob, gts*masks) + smoothLoss_cur1 + IOULoss(sal1_prob, gts*masks)
        else:
            sal_loss1 = ratio * structure_loss(sal1, gts)  + smoothLoss_cur1
        
        edges_gt = torch.sigmoid(sal2).detach()
        
        smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
        
        if opt.general_stage:
            sal_loss2 = ratio * CE(sal2_prob, gts*masks) + smoothLoss_cur2 + IOULoss(sal2_prob, gts*masks)
        else:
            sal_loss2 = ratio * structure_loss(sal2, gts) + smoothLoss_cur2

        img_edges = torch.sigmoid(img_edges)
        dep_edges = torch.sigmoid(dep_edges)
        
        edge_map1 = F.interpolate(edge_map, size=image_edges.size()[2:], mode='bilinear', align_corners=False)
        edge_map_depth1 = F.interpolate(edge_map_depth, size=depth_edges.size()[2:], mode='bilinear', align_corners=False)
        
        img_edges = F.interpolate(img_edges, size=image_edges.size()[2:], mode='bilinear', align_corners=False)
        dep_edges = F.interpolate(dep_edges, size=depth_edges.size()[2:], mode='bilinear', align_corners=False)
        
        image_edges = (image_edges > 0.5).float()
        depth_edges = (depth_edges > 0.5).float()
        

        edges_gt = label_edge_prediction(edges_gt)
        edge_loss = opt.edge_loss_weight * (structure_loss(edge_map1, edges_gt) + dice_loss(torch.sigmoid(edge_map1), image_edges))
        
        edge_loss += opt.edge_loss_weight * (structure_loss(edge_map_depth1, edges_gt) + dice_loss(torch.sigmoid(edge_map_depth1), depth_edges))

        loss = sal_loss1 + edge_loss + sal_loss2
        
        # if opt.vis_dir:
        #     if not os.path.exists(opt.vis_dir):
        #         os.makedirs(opt.vis_dir)
        #     visualize_prediction(torch.sigmoid(sal1), 'sal1', opt.vis_dir)
        #     visualize_prediction(torch.sigmoid(sal2), 'sal2', opt.vis_dir)
        #     visualize_prediction(torch.sigmoid(edge_map), 'edge', opt.vis_dir)
        #     visualize_prediction(torch.sigmoid(edge_map_depth), 'depth_edge', opt.vis_dir)


        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        if not opt.general_stage:
            teacher_model = teacher_model_update(teacher_model, model, opt.mom_coef)

        if i % 10 == 0 or i == total_step:
            if opt.general_stage:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data))
            else:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.format(
                        datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data))

    save_path = opt.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 5 == 0:
        if opt.general_stage:
            torch.save(model.state_dict(), os.path.join(save_path, 'NESS_general_{}.pth'.format(epoch)))
        else:
            torch.save(teacher_model.state_dict(), os.path.join(save_path, 'NESS_refinement_{}.pth'.format(epoch)))


def main():
    
    print("Scribble it! (general stage)") if opt.general_stage else print("Scribble it! (refinement stage)")
    print('Learning Rate: {}'.format(opt.lr))
    if opt.general_stage:
        model = NESS_Net(channel=32, general_stage=True)
        if (opt.load is not None):
            model.load_pre(opt.load)
            print('load model from ', opt.load)
        model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
        teacher_model = None
    else:
        model = NESS_Net(channel=32, general_stage=False)
        model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
        model.load_state_dict(torch.load(opt.general_model))
        teacher_model = NESS_Net(channel=32)
        
        if opt.keep_teacher:
            teacher_model = nn.DataParallel(teacher_model, device_ids=[0, 1])
            teacher_model = teacher_model.cuda(device=torch.device("cuda:0"))
            teacher_model.load_state_dict(torch.load(opt.general_model))
            for teacher_param in teacher_model.parameters():
                teacher_param.detach_()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    if opt.general_stage == 1:
        train_loader = get_loader(
            os.path.join(opt.data_dir, 'train_data', 'img'), os.path.join(opt.data_dir, 'train_data', 'depth'),
            os.path.join(opt.data_dir, 'train_data', 'gt'), os.path.join(opt.data_dir, 'train_data', 'mask'),
            os.path.join(opt.data_dir, 'train_data', 'gray'), os.path.join(opt.data_dir, 'train_data', 'rgb-lab'),
            os.path.join(opt.data_dir, 'train_data', 'depth-lab'), batchsize=opt.batchsize, 
            trainsize=opt.trainsize, general_stage=opt.general_stage
        )
    else:
        train_loader = get_loader(
            os.path.join(opt.data_dir, 'train_data', 'img'), os.path.join(opt.data_dir, 'train_data', 'depth'),
            os.path.join(opt.data_dir, 'train_data', 'gt_refinement'), os.path.join(opt.data_dir, 'train_data', 'mask'),
            os.path.join(opt.data_dir, 'train_data', 'gray'), os.path.join(opt.data_dir, 'train_data', 'rgb-lab'),
            os.path.join(opt.data_dir, 'train_data', 'depth-lab'), batchsize=opt.batchsize, 
            trainsize=opt.trainsize, general_stage=opt.general_stage
        )
        opt.epoch = 30
        
    if not opt.general_stage and not opt.keep_teacher:
        teacher_model = nn.DataParallel(teacher_model, device_ids=[0, 1])
        teacher_model = teacher_model.cuda(device=torch.device("cuda:0"))
        
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        if not opt.keep_teacher and teacher_model is not None:
            teacher_model.load_state_dict(torch.load(opt.general_model))
            for teacher_param in teacher_model.parameters():
                teacher_param.detach_()
        train(train_loader, model, teacher_model, optimizer, epoch)


if __name__ == '__main__':
    main()
