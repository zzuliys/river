import os
import torch

class Config:
    BASE_DIR = "/home/daiyin/data/code/river/kaggle_river/riwa_v2"
    TRAIN_IMAGES = os.path.join(BASE_DIR, "images")
    TRAIN_LABELS = os.path.join(BASE_DIR, "masks")

    VAL_IMAGES = os.path.join(BASE_DIR, "validation", "images") if os.path.exists(os.path.join(BASE_DIR, "validation", "images")) else os.path.join(BASE_DIR, "images")
    VAL_LABELS = os.path.join(BASE_DIR, "validation", "masks") if os.path.exists(os.path.join(BASE_DIR, "validation", "masks")) else os.path.join(BASE_DIR, "masks")

    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8

    LOG_DIR = os.path.join(os.getcwd(), "logs")
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
    RESULT_DIR = os.path.join(os.getcwd(), "results")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True

    IN_CHANNELS = 3
    NUM_CLASSES = 2

    USE_AMP = True
    AMP_dtype = torch.float16

    PREFETCH_FACTOR = 2

    USE_CHANNELS_LAST = True

    PERSISTENT_WORKERS = True