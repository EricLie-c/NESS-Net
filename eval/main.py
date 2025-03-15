import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset
def main(cfg):
    output_dir = cfg.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gt_dir = cfg.gt_root_dir
    pred_dir = cfg.pred_root_dir
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        #method_names = cfg.methods.split(' ')
        method_names = cfg.methods
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        #dataset_names = cfg.datasets.split(' ')
        dataset_names = cfg.datasets
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        print(thread.run())

if __name__ == "__main__":
    pred_root_dir='../results/'
    gt_root_dir='../dataset/test_data/gt/'
    MODEL_NAMES = ["NESS-Net"]
    MODEL_NAMES.sort(key=lambda x: x[:])
    # DATA_NAMES = os.listdir(gt_root_dir)#["ReDWeb-S"]
    # DATA_NAMES.sort(key=lambda x: x[:])
    DATA_NAMES = ["LFSD","NJU2K_Test"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=MODEL_NAMES)
    parser.add_argument('--datasets', nargs='+', default=DATA_NAMES)
    parser.add_argument('--gt_root_dir', type=str, default=gt_root_dir)
    parser.add_argument('--pred_root_dir', type=str, default=pred_root_dir)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
