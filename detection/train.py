import os
import torch
from ultralytics import YOLO

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(_PROJECT_DIR, "VOC2007_yolo", "data.yaml")

os.makedirs(os.path.join(_PROJECT_DIR, "runs"), exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data: {DATA_YAML}")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=416,
        batch=32,
        device=device,
        workers=4,
        project=os.path.join(_PROJECT_DIR, "runs"),
        name="detect_train",
        exist_ok=True,
        amp=True,
        cos_lr=True,
        warmup_epochs=3,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer="auto",
        patience=30,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n训练完成! 最佳模型: {results.save_dir}/weights/best.pt")

    metrics = model.val(data=DATA_YAML, split="test", imgsz=416, batch=32, device=device)
    print(f"\n测试集结果 - mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
