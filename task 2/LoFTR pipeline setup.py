import os
from glob import glob
import time
import matplotlib.pyplot as plt
import gc
import sys
import cv2
from typing import Dict, Optional
import argparse
import rasterio
from rasterio.plot import reshape_as_image
import kornia as K
import kornia.feature as KF
import kornia_moons.feature as KMF

import numpy as np
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *

import torch
import torch.nn as nn

import torch

class LoFTRMatcher:
    def __init__(self, device=None, input_longside=1200, conf_th=None):
        self.device = device
        self.matcher = K.feature.LoFTR(pretrained='outdoor').eval().to(self.device)
        self.input_longside = input_longside
        self.conf_thresh = conf_th
        
    def prep_img(self, img, long_side=1200):
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1]) 
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        else:
            scale = 1.0

        img_ts = K.image_to_tensor(img, False).float() / 255.
        img_ts = K.color.bgr_to_rgb(img_ts)
        img_ts = K.color.rgb_to_grayscale(img_ts)
        return img, img_ts.to(self.device), scale
    
    def tta_rotation_preprocess(self, img_np, angle):
        rot_M = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1)
        rot_M_inv = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1)
        rot_img = cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))

        rot_img_ts = K.image_to_tensor(rot_img, False).float() / 255.
        rot_img_ts = K.color.bgr_to_rgb(rot_img_ts)
        rot_img_ts = K.color.rgb_to_grayscale(rot_img_ts)
        return rot_M, rot_img_ts.to(self.device), rot_M_inv

    def tta_rotation_postprocess(self, kpts, img_np, rot_M_inv):
        ones = np.ones(shape=(kpts.shape[0], ), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < img_np.shape[1]) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < img_np.shape[0])
        return rot_kpts, mask

    def __call__(self, img_np1, img_np2, tta=['orig']):
        with torch.no_grad():
            img_np1, img_ts0, scale0 = self.prep_img(img_np1, long_side=self.input_longside)
            img_np2, img_ts1, scale1 = self.prep_img(img_np2, long_side=self.input_longside)
            images0, images1 = [], []

            # TTA
            for tta_elem in tta:
                if tta_elem == 'orig':
                    img_ts0_aug, img_ts1_aug = img_ts0, img_ts1
                elif tta_elem == 'flip_lr':
                    img_ts0_aug = torch.flip(img_ts0, [3, ])
                    img_ts1_aug = torch.flip(img_ts1, [3, ])
                elif tta_elem == 'flip_ud':
                    img_ts0_aug = torch.flip(img_ts0, [2, ])
                    img_ts1_aug = torch.flip(img_ts1, [2, ])
                elif tta_elem == 'rot_r10':
                    rot_r10_M0, img_ts0_aug, rot_r10_M0_inv = self.tta_rotation_preprocess(img_np1, 10)
                    rot_r10_M1, img_ts1_aug, rot_r10_M1_inv = self.tta_rotation_preprocess(img_np2, 10)
                elif tta_elem == 'rot_l10':
                    rot_l10_M0, img_ts0_aug, rot_l10_M0_inv = self.tta_rotation_preprocess(img_np1, -10)
                    rot_l10_M1, img_ts1_aug, rot_l10_M1_inv = self.tta_rotation_preprocess(img_np2, -10)
                else:
                    raise ValueError('Unknown TTA method.')
                images0.append(img_ts0_aug)
                images1.append(img_ts1_aug)

            # Inference
            input_dict = {"image0": torch.cat(images0), "image1": torch.cat(images1)}
            correspondences = self.matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            batch_id = correspondences['batch_indexes'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            # Reverse TTA
            for idx, tta_elem in enumerate(tta):
                batch_mask = batch_id == idx

                if tta_elem == 'orig':
                    pass
                elif tta_elem == 'flip_lr':
                    mkpts0[batch_mask, 0] = img_np1.shape[1] - mkpts0[batch_mask, 0]
                    mkpts1[batch_mask, 0] = img_np2.shape[1] - mkpts1[batch_mask, 0]
                elif tta_elem == 'flip_ud':
                    mkpts0[batch_mask, 1] = img_np1.shape[0] - mkpts0[batch_mask, 1]
                    mkpts1[batch_mask, 1] = img_np2.shape[0] - mkpts1[batch_mask, 1]
                elif tta_elem == 'rot_r10':
                    mkpts0[batch_mask], mask0 = self.tta_rotation_postprocess(mkpts0[batch_mask], img_np1, rot_r10_M0_inv)
                    mkpts1[batch_mask], mask1 = self.tta_rotation_postprocess(mkpts1[batch_mask], img_np2, rot_r10_M1_inv)
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(np.float32) * -10.
                elif tta_elem == 'rot_l10':
                    mkpts0[batch_mask], mask0 = self.tta_rotation_postprocess(mkpts0[batch_mask], img_np1, rot_l10_M0_inv)
                    mkpts1[batch_mask], mask1 = self.tta_rotation_postprocess(mkpts1[batch_mask], img_np2, rot_l10_M1_inv)
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(np.float32) * -10.
                else:
                    raise ValueError('Unknown TTA method.')
                    
            if self.conf_thresh is not None:
                th_mask = confidence >= self.conf_thresh
            else:
                th_mask = confidence >= 0.
            mkpts0, mkpts1 = mkpts0[th_mask, :], mkpts1[th_mask, :]

            # Matching points
            return mkpts0 / scale0, mkpts1 / scale1


# Helper to read raster images
def read_raster_image(path):
    with rasterio.open(path, "r", driver='JP2OpenJPEG') as src:
        raster_image = src.read()
        raster_meta = src.meta
    raster_image = reshape_as_image(raster_image)
    return raster_image, raster_meta

# Image matching and processing class
class SolutionHolder:
    def __init__(self):
        self.F_dict = dict()

    @staticmethod
    def solve_keypoints(mkpts0, mkpts1):
        if len(mkpts0) > 7:
            F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.9999, 250000)
            inliers = inliers > 0
        else:
            F, inliers = np.zeros((3, 3)), None
        return F, inliers

    def add_solution(self, sample_id, mkpts0, mkpts1):
        self.F_dict[sample_id] = SolutionHolder.solve_keypoints(mkpts0, mkpts1)

# Main function
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_folder = '../input/deforestation-in-ukraine'
    images_num = 0
    image_names_dict = {}

    for dirname, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith('_TCI.jp2'):
                images_num += 1
                if dirname not in image_names_dict:
                    image_names_dict[dirname] = list()
                image_names_dict[dirname].append(filename)

    image_paths_list = [os.path.join(dirname, filename) for dirname, filenames in image_names_dict.items() for filename in filenames]

    read_image_indexes = [0, 2, 14, 19]
    images = []

    for i in read_image_indexes:
        image_path = image_paths_list[i]
        image, _ = read_raster_image(image_path)
        images.append(image)

    TARGET_SIZE = (1098, 1098)
    dscale_images = [cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA) for img in images]

    loftr_matcher = LoFTRMatcher(device=device, input_longside=1200, conf_th=0.3)

    image_pairs = list(zip([0, 2, 19, 0], [19, 14, 14, 2]))
    
    for idx, (i1, i2) in enumerate(image_pairs):
        print(f"\nProcessing Pair {idx + 1}: Image {i1} and Image {i2}")
        img1, img2 = dscale_images[read_image_indexes.index(i1)], dscale_images[read_image_indexes.index(i2)]
        mkpts0, mkpts1 = loftr_matcher(img1, img2)
        F, inliers = SolutionHolder.solve_keypoints(mkpts0, mkpts1)

        KMF.draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                         torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                         torch.ones(mkpts0.shape[0]).view(1,-1, 1)),
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                         torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                         torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
            cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None, 
                       'feature_color': (0.2, 0.5, 1), 'vertical': False},
        )

        gc.collect()
        print("Processing complete.")

