import os
import cv2
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RiverDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        t0 = time.time()
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        label_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        label_path = os.path.join(self.labels_dir, label_name)
        t1 = time.time()

        img = cv2.imread(img_path)
        t2 = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t3 = time.time()

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        t4 = time.time()

        if label is None:
            label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        t5 = time.time()

        label = (label > 127).astype(np.uint8) * 255
        t6 = time.time()

        if self.transform:
            img = self.transform(img)
        t7 = time.time()

        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        t8 = time.time()
        label = torch.from_numpy(label).long() // 255
        t9 = time.time()

        total_time = t9 - t0
        print(f"[{idx}] {img_name}:")
        print(f"  - 路径构建: {(t1-t0)*1000:.2f}ms")
        print(f"  - 读取图像: {(t2-t1)*1000:.2f}ms")
        print(f"  - 颜色转换: {(t3-t2)*1000:.2f}ms")
        print(f"  - 读取标签: {(t4-t3)*1000:.2f}ms")
        print(f"  - 标签检查: {(t5-t4)*1000:.2f}ms")
        print(f"  - 标签二值化: {(t6-t5)*1000:.2f}ms")
        print(f"  - 图像变换: {(t7-t6)*1000:.2f}ms")
        print(f"  - 标签缩放: {(t8-t7)*1000:.2f}ms")
        print(f"  - 标签转张量: {(t9-t8)*1000:.2f}ms")
        print(f"  - TOTAL: {total_time*1000:.2f}ms")

        return img, label

def get_dataloader(images_dir, labels_dir, batch_size, num_workers, image_size=512, shuffle=True):
    dataset = RiverDataset(images_dir, labels_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)