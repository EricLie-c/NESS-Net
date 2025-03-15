import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.NESS_Net import NESS_Net

from data import test_dataset

import pydensecrf.densecrf as dcrf

os.environ['CUDA_VISIBLE_DEVICES'] = '6,4'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--general_stage', action='store_true', help='general stage without consistency training')
parser.add_argument('--second_round', action='store_true', help='second-round model')
parser.add_argument('--data_dir', default='./dataset', help='path where to save testing data')
parser.add_argument('--model_path', default='./models/teacher_scribble_30.pth', help='path where to save trained model')
parser.add_argument('--save_dir', default='./dataset/train_data', help='path where to save predicted maps')
opt = parser.parse_args()

SEED = 42
torch.manual_seed(SEED)   
torch.cuda.manual_seed(SEED)          
torch.cuda.manual_seed_all(SEED)      
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False     

model = NESS_Net(channel=32, general_stage=opt.general_stage or opt.second_round)
device = torch.device("cuda:0")
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
model.load_state_dict(torch.load(opt.model_path))
model.eval()

save_path = os.path.join(opt.save_dir, 'gt_refinement')
if not os.path.exists(save_path):
    os.makedirs(save_path)
image_root = os.path.join(opt.data_dir, 'train_data', 'img')
depth_root = os.path.join(opt.data_dir, 'train_data', 'depth')
rgb_edge_root = os.path.join(opt.data_dir, 'train_data', 'rgb-lab')
depth_edge_root = os.path.join(opt.data_dir, 'train_data', 'depth-lab')


def crf_refine(img, annos):      #use crf to refine predict pic
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32') # set a U which is the same size as input pic
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])  # the same size with the input pic
    return res.astype('uint8')


test_loader = test_dataset(image_root, depth_root, rgb_edge_root, depth_edge_root, opt.testsize)
for i in tqdm(range(test_loader.size)):
    image, depth, HH, WW, name, image_edge, depth_edge = test_loader.load_data()
    
    _, _, res, _, _, _ = model(image.to(device), depth.to(device), image_edge.to(device), depth_edge.to(device))

    
    res = F.interpolate(res, size=[WW,HH], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().squeeze()
    res = np.ascontiguousarray(res)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = np.round(res * 255).astype(np.uint8)
    
    image = F.interpolate(image, size=[WW,HH], mode='bilinear', align_corners=False)
    img = image.sigmoid().data.cpu().squeeze().permute(1,2,0)
    img = np.ascontiguousarray(img)
    img = np.round(img).astype(np.uint8)

    res = crf_refine(img, res)
    
    cv2.imwrite(os.path.join(save_path, name), res)