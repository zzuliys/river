import os
import cv2
import numpy as np
import torch
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
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        label_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        label_path = os.path.join(self.labels_dir, label_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label is None:
            label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        label = (label > 127).astype(np.uint8) * 255

        if self.transform:
            img = self.transform(img)

        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label).long() // 255
        return img, label


class PreprocessedDataset(Dataset):
    def __init__(self, dataset_dir, mean, std):
        self.img_dir = os.path.join(dataset_dir, "images")
        self.label_dir = os.path.join(dataset_dir, "labels")
        self.files = sorted(os.listdir(self.img_dir))
        self._len = len(self.files)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)

        label = cv2.imread(os.path.join(self.label_dir, fname), cv2.IMREAD_GRAYSCALE)
        label = torch.from_numpy(label).long() // 255
        return img, label

def get_dataloader(images_dir, labels_dir, batch_size, num_workers, image_size=512, shuffle=True, prefetch_factor=2, persistent_workers=True):
    dataset = RiverDataset(images_dir, labels_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)