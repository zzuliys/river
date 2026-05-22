import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def xywh2xyxy_np(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh_np(x):
    y = x.copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def letterbox(img, new_shape=416, color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = r
    pad = (dw, dh)
    return img, ratio, pad


def random_affine(img, targets=(), degrees=10, translate=0.1, scale=0.5, shear=10, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * height
    T[1, 2] = random.uniform(-translate, translate) * height

    M = T @ R
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        xy = targets[:, 1:].copy()
        xy[:, :2] = xy[:, :2] * width - border[0]
        xy[:, 2:4] = xy[:, 2:4] * height - border[1]

        xy_homog = np.ones((n, 3))
        xy_homog[:, :2] = xy[:, :2]
        xy_homog = xy_homog @ M.T
        xy_homog[:, :2] /= width

        wh_scaled = xy[:, 2:4] * s / height

        targets[:, 1:3] = xy_homog[:, :2]
        targets[:, 3:5] = wh_scaled

        targets = targets[targets[:, 3] > 0.001]
        targets = targets[targets[:, 4] > 0.001]

    return img, targets


def hsv_augment(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


class YOLODataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=416, augment=True, stride=32):
        self.img_size = img_size
        self.augment = augment
        self.stride = stride

        img_dir = os.path.join(data_dir, "images", split)
        label_dir = os.path.join(data_dir, "labels", split)

        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith('.txt')
        ])

    def __len__(self):
        return len(self.img_paths)

    def load_image(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, (img.shape[0], img.shape[1])

    def load_labels(self, idx):
        path = self.label_paths[idx]
        labels = []
        if os.path.getsize(path):
            labels = np.loadtxt(path, ndmin=2)
            if len(labels) == 0:
                labels = np.zeros((0, 5))
        else:
            labels = np.zeros((0, 5))
        return labels.reshape(-1, 5)

    def __getitem__(self, idx):
        img, (h0, w0) = self.load_image(idx)
        labels = self.load_labels(idx)

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

        img, ratio, pad = letterbox(img, self.img_size, stride=self.stride)

        if len(labels):
            labels[:, 1] = labels[:, 1] * w0 * ratio + pad[0]
            labels[:, 2] = labels[:, 2] * h0 * ratio + pad[1]
            labels[:, 3] = labels[:, 3] * w0 * ratio
            labels[:, 4] = labels[:, 4] * h0 * ratio
            labels[:, 1:] /= self.img_size

        if self.augment:
            img, labels = random_affine(img, labels)
            img = hsv_augment(img)

        img = img.astype(np.float32) / 255.0

        nl = len(labels)
        targets = np.zeros((nl, 6), dtype=np.float32)
        if nl:
            targets[:, 0] = idx
            targets[:, 1] = labels[:, 0]
            targets[:, 2:] = labels[:, 1:5]

        img = torch.from_numpy(img).permute(2, 0, 1)
        targets = torch.from_numpy(targets)

        return img, targets

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch], 0)
        targets = [item[1] for item in batch]
        for i, t in enumerate(targets):
            if len(t):
                t[:, 0] = i
        targets = torch.cat(targets, 0) if len(targets) and any(len(t) for t in targets) else torch.zeros((0, 6))
        return imgs, targets
