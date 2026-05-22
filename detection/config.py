import os
import torch

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- 数据 ---
    DATA_DIR = os.path.join(_PROJECT_DIR, "VOC2007_yolo")

    IMAGE_SIZE = 416
    BATCH_SIZE = 32 if torch.cuda.is_available() else 4
    NUM_WORKERS = 0
    PIN_MEMORY = True

    # --- 模型 ---
    IN_CHANNELS = 3
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "ball", "bottle", "branch", "grass",
        "leaf", "milk-box", "plastic-bag", "plastic-garbage",
    ]
    ANCHORS = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    STRIDES = [8, 16, 32]

    # --- 损失 ---
    BOX_GAIN = 7.5
    CLS_GAIN = 2.0
    OBJ_GAIN = 0.7

    # --- 训练 ---
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0005
    COS_LR = True

    # --- 评估 ---
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.5
    MAP_EVAL_INTERVAL = 5

    # --- 路径 ---
    CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, "checkpoints")
    LOG_DIR = os.path.join(_PROJECT_DIR, "logs")
    PLOTS_DIR = os.path.join(_PROJECT_DIR, "plots")

    # --- 设备 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 性能 ---
    USE_AMP = torch.cuda.is_available()
    AMP_DTYPE = torch.float16
    CUDNN_BENCHMARK = True

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)
