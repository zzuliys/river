import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config

PREPROC_DIR = Config.PREPROC_DIR
TRAIN_IMAGES_DIR = Config.TRAIN_IMAGES
TRAIN_LABELS_DIR = Config.TRAIN_LABELS
VAL_IMAGES_DIR = Config.VAL_IMAGES
VAL_LABELS_DIR = Config.VAL_LABELS
IMAGE_SIZE = Config.IMAGE_SIZE


def compute_dataset_stats(image_dir, max_samples=None):
    print(f"Computing mean/std from: {image_dir} ...")
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    if max_samples:
        files = files[:max_samples]

    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    count = 0

    for fname in tqdm(files, desc="Scanning stats"):
        img = cv2.imread(os.path.join(image_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean_sum += img.mean(axis=(0, 1))
        std_sum += img.std(axis=(0, 1))
        count += 1

    mean = mean_sum / count
    std = std_sum / count
    print(f"Dataset mean: {mean.tolist()}, std: {std.tolist()}")
    return mean, std


def process_one_image(args):
    idx, fname, image_dir, label_dir, out_img_dir, out_label_dir = args
    img_path = os.path.join(image_dir, fname)

    label_name = fname.replace('.jpg', '.png').replace('.JPG', '.png')
    label_path = os.path.join(label_dir, label_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = (label > 127).astype(np.uint8) * 255
    label = cv2.resize(label, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    out_name = f"{idx:06d}.png"
    cv2.imwrite(os.path.join(out_img_dir, out_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_label_dir, out_name), label)
    return True


def preprocess_split(name, image_dir, label_dir, num_workers=8):
    out_img_dir = os.path.join(PREPROC_DIR, name, "images")
    out_label_dir = os.path.join(PREPROC_DIR, name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Preprocessing {name}: {len(files)} images ...")

    args_list = [(i, f, image_dir, label_dir, out_img_dir, out_label_dir)
                 for i, f in enumerate(files)]

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_one_image, a) for a in args_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {name}"):
            pass

    print(f"{name} done: {len(files)} samples.")


def main():
    if os.path.exists(PREPROC_DIR):
        print(f"Preprocessed dir already exists: {PREPROC_DIR}")
        print("Delete it first to re-preprocess, or just use it as-is.")
        return

    mean, std = compute_dataset_stats(TRAIN_IMAGES_DIR)
    preprocess_split("train", TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    preprocess_split("val", VAL_IMAGES_DIR, VAL_LABELS_DIR)

    meta_path = os.path.join(PREPROC_DIR, "meta.pt")
    torch.save({
        'mean': mean.tolist(),
        'std': std.tolist(),
        'image_size': IMAGE_SIZE,
    }, meta_path)
    print(f"\nAll done! Preprocessed data at: {PREPROC_DIR}")
    print(f"mean: {mean.tolist()}, std: {std.tolist()}")


if __name__ == "__main__":
    main()
