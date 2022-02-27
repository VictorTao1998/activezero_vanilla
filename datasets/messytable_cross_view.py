"""
Author: Isabella Liu 2/26/22
Feature: Load data from messy-table-dataset, and load temporal IR pattern
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
import cv2

from utils.config import cfg
from utils.util import load_pickle


def __gamma_trans__(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def __get_ir_pattern__(img_ir: np.array, img: np.array, threshold=0.005):
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    ir = np.zeros_like(diff)
    ir[diff > threshold] = 1
    return ir


def __get_smoothed_ir_pattern__(img_ir: np.array, img: np.array, ks=11):
    h, w = img_ir.shape
    hs = int(h//ks)
    ws = int(w//ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws,hs), interpolation = cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w,h), interpolation = cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    ir[diff > diff_avg] = 1
    return ir


def __get_smoothed_ir_pattern2__(img_ir: np.array, img: np.array, ks=11, threshold=0.005):
    h, w = img_ir.shape
    hs = int(h//ks)
    ws = int(w//ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws,hs), interpolation = cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w,h), interpolation = cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    diff2 = diff - diff_avg
    ir[diff2 > threshold] = 1
    return ir


def __data_augmentation__(gaussian_blur=False, color_jitter=False):
    """
    :param gaussian_blur: Whether apply gaussian blur in data augmentation
    :param color_jitter: Whether apply color jitter in data augmentation
    Note:
        If you want to change the parameters of each augmentation, you need to go to config files,
        e.g. configs/remote_train_config.yaml
    """
    transform_list = [
        Transforms.ToTensor()
    ]
    if gaussian_blur:
        gaussian_sig = random.uniform(cfg.DATA_AUG.GAUSSIAN_MIN, cfg.DATA_AUG.GAUSSIAN_MAX)
        transform_list += [
            Transforms.GaussianBlur(kernel_size=cfg.DATA_AUG.GAUSSIAN_KERNEL, sigma=gaussian_sig)
        ]
    if color_jitter:
        bright = random.uniform(cfg.DATA_AUG.BRIGHT_MIN, cfg.DATA_AUG.BRIGHT_MAX)
        contrast = random.uniform(cfg.DATA_AUG.CONTRAST_MIN, cfg.DATA_AUG.CONTRAST_MAX)
        transform_list += [
            Transforms.ColorJitter(brightness=[bright, bright],
                                   contrast=[contrast, contrast])
        ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


def __get_split_files__(split_file, debug=False, sub=100):
    """
    :param split_file: Path to the split .txt file, e.g. train.txt
    :param debug: Debug mode, load less data
    :param sub: If debug mode is enabled, sub will be the number of data loaded
    :param onReal: Whether test on real dataset, folder and file names are different
    :return: Lists of paths to the entries listed in split file
    """
    # Sim
    with open(split_file, 'r') as f:
        prefix = [line.strip().split(',') for line in f]
        np.random.shuffle(prefix)

    img_L1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.LEFT) for p in prefix]
    img_R1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.RIGHT) for p in prefix]
    img_L_no_ir1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.LEFT_NO_IR) for p in prefix]
    img_R_no_ir1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.RIGHT_NO_IR) for p in prefix]
    img_depth_l1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.DEPTHL) for p in prefix]
    img_depth_r1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.DEPTHR) for p in prefix]
    img_meta1 = [os.path.join(cfg.DIR.DATASET, p[0], cfg.SPLIT.META) for p in prefix]
    img_label1 = [os.path.join(cfg.REAL.DATASET, p[0], cfg.SPLIT.LABEL) for p in prefix]
    
    img_L2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.LEFT) for p in prefix]
    img_R2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.RIGHT) for p in prefix]
    img_L_no_ir2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.LEFT_NO_IR) for p in prefix]
    img_R_no_ir2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.RIGHT_NO_IR) for p in prefix]
    img_depth_l2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.DEPTHL) for p in prefix]
    img_depth_r2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.DEPTHR) for p in prefix]
    img_meta2 = [os.path.join(cfg.DIR.DATASET, p[1], cfg.SPLIT.META) for p in prefix]
    img_label2 = [os.path.join(cfg.REAL.DATASET, p[1], cfg.SPLIT.LABEL) for p in prefix]

    if debug is True:
        img_L1, img_L2 = img_L1[:sub], img_L2[:sub]
        img_R1, img_R2 = img_R1[:sub], img_R2[:sub]
        img_L_no_ir1, img_L_no_ir2 = img_L_no_ir1[:sub], img_L_no_ir2[:sub]
        img_R_no_ir1, img_R_no_ir2 = img_R_no_ir1[:sub], img_R_no_ir2[:sub]
        img_depth_l1, img_depth_l2 = img_depth_l1[:sub], img_depth_l2[:sub]
        img_depth_r1, img_depth_r2 = img_depth_r1[:sub], img_depth_r2[:sub]
        img_meta1, img_meta2 = img_meta1[:sub], img_meta2[:sub]
        img_label1, img_label2 = img_label1[:sub], img_label2[:sub]

    return img_L1, img_R1, img_L_no_ir1, img_R_no_ir1, img_depth_l1, img_depth_r1, img_meta1, img_label1, \
           img_L2, img_R2, img_L_no_ir2, img_R_no_ir2, img_depth_l2, img_depth_r2, img_meta2, img_label2


class MessytableDataset(Dataset):
    def __init__(self, split_file, gaussian_blur=False, color_jitter=False, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param gaussian_blur: Whether apply gaussian blur in data augmentation
        :param color_jitter: Whether apply color jitter in data augmentation
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L1, self.img_R1, self.img_L_no_ir1, self.img_R_no_ir1, self.img_depth_l1, self.img_depth_r1, \
        self.img_meta1, self.img_label1, \
        self.img_L2, self.img_R2, self.img_L_no_ir2, self.img_R_no_ir2, self.img_depth_l2, self.img_depth_r2, \
        self.img_meta2, self.img_label2 = __get_split_files__(split_file, debug, sub)

        self.gaussian_blur = gaussian_blur
        self.color_jitter = color_jitter

    def __len__(self):
        return len(self.img_L1)

    def __getitem__(self, idx):
        # Scene 1
        img_L1 = np.array(Image.open(self.img_L1[idx]).convert(mode='L')) / 255  # [H, W]
        img_R1 = np.array(Image.open(self.img_R1[idx]).convert(mode='L')) / 255
        img_L_no_ir1 = np.array(Image.open(self.img_L_no_ir1[idx]).convert(mode='L')) / 255
        img_R_no_ir1 = np.array(Image.open(self.img_R_no_ir1[idx]).convert(mode='L')) / 255
        img_L_ir_pattern1 = __get_smoothed_ir_pattern2__(img_L1, img_L_no_ir1)  # [H, W]
        img_R_ir_pattern1 = __get_smoothed_ir_pattern2__(img_R1, img_R_no_ir1)
        img_L_rgb1 = np.repeat(img_L1[:, :, None], 3, axis=-1)
        img_R_rgb1 = np.repeat(img_R1[:, :, None], 3, axis=-1)

        img_depth_l1 = np.array(Image.open(self.img_depth_l1[idx])) / 1000  # convert from mm to m
        img_depth_r1 = np.array(Image.open(self.img_depth_r1[idx])) / 1000  # convert from mm to m
        img_meta1 = load_pickle(self.img_meta1[idx])

        # Cam params
        cam_intrinsic1 = img_meta1['intrinsic_l']
        cam_intrinsic1[:2] /= 2
        cam_extrinsic1 = img_meta1['extrinsic_l']

        # Convert depth map to disparity map
        extrinsic_l1 = img_meta1['extrinsic_l']
        extrinsic_r1 = img_meta1['extrinsic_r']
        intrinsic_l1 = img_meta1['intrinsic_l']
        baseline1 = np.linalg.norm(extrinsic_l1[:, -1] - extrinsic_r1[:, -1])
        focal_length1 = intrinsic_l1[0, 0] / 2

        mask = img_depth_l1 > 0
        img_disp_l1 = np.zeros_like(img_depth_l1)
        img_disp_l1[mask] = focal_length1 * baseline1 / img_depth_l1[mask]
        mask = img_depth_r1 > 0
        img_disp_r1 = np.zeros_like(img_depth_r1)
        img_disp_r1[mask] = focal_length1 * baseline1 / img_depth_r1[mask]

        # random crop the image to CROP_HEIGHT * CROP_WIDTH
        h, w = img_L_rgb1.shape[:2]
        th, tw = cfg.ARGS.CROP_HEIGHT, cfg.ARGS.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        img_L_rgb1 = img_L_rgb1[x:(x + th), y:(y + tw)]
        img_R_rgb1 = img_R_rgb1[x:(x + th), y:(y + tw)]
        img_L_ir_pattern1 = img_L_ir_pattern1[x:(x + th), y:(y + tw)]
        img_R_ir_pattern1 = img_R_ir_pattern1[x:(x + th), y:(y + tw)]
        img_disp_l1 = img_disp_l1[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]  # depth original res in 1080*1920
        img_depth_l1 = img_depth_l1[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]
        img_disp_r1 = img_disp_r1[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]
        img_depth_r1 = img_depth_r1[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]

        # Scene 2
        img_L2 = np.array(Image.open(self.img_L2[idx]).convert(mode='L')) / 255  # [H, W]
        img_R2 = np.array(Image.open(self.img_R2[idx]).convert(mode='L')) / 255
        img_L_no_ir2 = np.array(Image.open(self.img_L_no_ir2[idx]).convert(mode='L')) / 255
        img_R_no_ir2 = np.array(Image.open(self.img_R_no_ir2[idx]).convert(mode='L')) / 255
        img_L_ir_pattern2 = __get_smoothed_ir_pattern2__(img_L2, img_L_no_ir2)  # [H, W]
        img_R_ir_pattern2 = __get_smoothed_ir_pattern2__(img_R2, img_R_no_ir2)
        img_L_rgb2 = np.repeat(img_L2[:, :, None], 3, axis=-1)
        img_R_rgb2 = np.repeat(img_R2[:, :, None], 3, axis=-1)

        img_depth_l2 = np.array(Image.open(self.img_depth_l2[idx])) / 1000  # convert from mm to m
        img_depth_r2 = np.array(Image.open(self.img_depth_r2[idx])) / 1000  # convert from mm to m
        img_meta2 = load_pickle(self.img_meta2[idx])

        # Cam params
        cam_intrinsic2 = img_meta2['intrinsic_l']
        cam_intrinsic2[:2] /= 2
        cam_extrinsic2 = img_meta2['extrinsic_l']

        # Convert depth map to disparity map
        extrinsic_l2 = img_meta2['extrinsic_l']
        extrinsic_r2 = img_meta2['extrinsic_r']
        intrinsic_l2 = img_meta2['intrinsic_l']
        baseline2 = np.linalg.norm(extrinsic_l2[:, -1] - extrinsic_r2[:, -1])
        focal_length2 = intrinsic_l2[0, 0] / 2

        mask = img_depth_l2 > 0
        img_disp_l2 = np.zeros_like(img_depth_l2)
        img_disp_l2[mask] = focal_length2 * baseline2 / img_depth_l2[mask]
        mask = img_depth_r2 > 0
        img_disp_r2 = np.zeros_like(img_depth_r2)
        img_disp_r2[mask] = focal_length2 * baseline2 / img_depth_r2[mask]

        # random crop the image to CROP_HEIGHT * CROP_WIDTH
        h, w = img_L_rgb2.shape[:2]
        th, tw = cfg.ARGS.CROP_HEIGHT, cfg.ARGS.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        img_L_rgb2 = img_L_rgb2[x:(x + th), y:(y + tw)]
        img_R_rgb2 = img_R_rgb2[x:(x + th), y:(y + tw)]
        img_L_ir_pattern2 = img_L_ir_pattern2[x:(x + th), y:(y + tw)]
        img_R_ir_pattern2 = img_R_ir_pattern2[x:(x + th), y:(y + tw)]
        img_disp_l2 = img_disp_l2[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]  # depth original res in 1080*1920
        img_depth_l2 = img_depth_l2[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]
        img_disp_r2 = img_disp_r2[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]
        img_depth_r2 = img_depth_r2[2 * x: 2 * (x + th), 2 * y: 2 * (y + tw)]


        # Get data augmentation
        custom_augmentation = __data_augmentation__(gaussian_blur=self.gaussian_blur, color_jitter=self.color_jitter)
        normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item['img_L1'] = custom_augmentation(img_L_rgb1).type(torch.FloatTensor)
        item['img_R1'] = custom_augmentation(img_R_rgb1).type(torch.FloatTensor)
        item['img_L_ir_pattern1'] = torch.tensor(img_L_ir_pattern1, dtype=torch.float32).unsqueeze(0)
        item['img_R_ir_pattern1'] = torch.tensor(img_R_ir_pattern1, dtype=torch.float32).unsqueeze(0)
        item['img_disp_l1'] = torch.tensor(img_disp_l1, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l1'] = torch.tensor(img_depth_l1, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r1'] = torch.tensor(img_disp_r1, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r1'] = torch.tensor(img_depth_r1, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix1'] = self.img_L1[idx].split('/')[-2]
        item['focal_length1'] = torch.tensor(focal_length1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline1'] = torch.tensor(baseline1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['cam_intrinsic1'] = torch.from_numpy(cam_intrinsic1).type(torch.float64)  # [bs, 3, 3]
        item['cam_extrinsic1'] = torch.from_numpy(cam_extrinsic1).type(torch.float64)  # [bs, 4, 4]

        item['img_L2'] = custom_augmentation(img_L_rgb2).type(torch.FloatTensor)
        item['img_R2'] = custom_augmentation(img_R_rgb2).type(torch.FloatTensor)
        item['img_L_ir_pattern2'] = torch.tensor(img_L_ir_pattern2, dtype=torch.float32).unsqueeze(0)
        item['img_R_ir_pattern2'] = torch.tensor(img_R_ir_pattern2, dtype=torch.float32).unsqueeze(0)
        item['img_disp_l2'] = torch.tensor(img_disp_l2, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W] in dataloader
        item['img_depth_l2'] = torch.tensor(img_depth_l2, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r2'] = torch.tensor(img_disp_r2, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r2'] = torch.tensor(img_depth_r2, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix2'] = self.img_L2[idx].split('/')[-2]
        item['focal_length2'] = torch.tensor(focal_length2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline2'] = torch.tensor(baseline2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['cam_intrinsic2'] = torch.from_numpy(cam_intrinsic2).type(torch.float64)  # [bs, 3, 3]
        item['cam_extrinsic2'] = torch.from_numpy(cam_extrinsic2).type(torch.float64)  # [bs, 4, 4]

        return item


if __name__ == '__main__':
    cdataset = MessytableDataset('/code/dataset_local_v9/training_lists/cross_view_train.txt')
    item = cdataset.__getitem__(1)
    print(item['img_L1'].shape)
    print(item['img_L2'].shape)

    print(item['cam_intrinsic1'].shape)
    print(item['cam_intrinsic2'].shape)

    print(item['cam_extrinsic1'].shape)
    print(item['cam_extrinsic2'].shape)
    # print(item['img_R'].shape)
    # print(item['img_disp_l'].shape)
    print(item['prefix1'])
    print(item['prefix2'])
    # print(item['img_real_L'].shape)
    # print(item['img_L_ir_pattern'].shape)
    # print(item['img_real_L_ir_pattern'].shape)
