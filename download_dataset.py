import os
import sys
import shutil


DATASET_SLUG = "franzwagner/river-water-segmentation-dataset"
LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_river")
LOCAL_RIWA = os.path.join(LOCAL_ROOT, "riwa_v2")


def is_dataset_ready():
    images_dir = os.path.join(LOCAL_RIWA, "images")
    masks_dir = os.path.join(LOCAL_RIWA, "masks")
    return os.path.isdir(images_dir) and os.path.isdir(masks_dir)


def download_dataset():
    if is_dataset_ready():
        print(f"Dataset already exists: {LOCAL_RIWA}")
        return LOCAL_RIWA

    print(f"Downloading dataset: {DATASET_SLUG} ...")
    import kagglehub
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"KaggleHub cache: {cache_path}")

    if os.path.exists(LOCAL_ROOT):
        if os.path.islink(LOCAL_ROOT):
            os.unlink(LOCAL_ROOT)
        else:
            shutil.rmtree(LOCAL_ROOT)

    parent = os.path.dirname(LOCAL_ROOT)
    os.makedirs(parent, exist_ok=True)
    os.symlink(cache_path, LOCAL_ROOT)
    print(f"Symlink created: {LOCAL_ROOT} -> {cache_path}")
    print(f"Dataset ready: {LOCAL_RIWA}")
    return LOCAL_RIWA


if __name__ == "__main__":
    download_dataset()
