import os
import random

import cv2
import numpy as np
import torch

from config import Config
from model import YOLONet
from utils import process_predictions, _filter_dets
from dataset import letterbox

Config.ensure_dirs()

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 0, 255), (255, 128, 0),
]


def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        # checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best.pt")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(Config.DEVICE)
    model = YOLONet(nc=Config.NUM_CLASSES, anchors=Config.ANCHORS)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"mAP@50: {ckpt.get('mAP50', 0.0):.4f})")
    return model, device


def preprocess(img_bgr, img_size=None):
    if img_size is None:
        img_size = Config.IMAGE_SIZE
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized, ratio, pad = letterbox(img_rgb, img_size, stride=32)
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, (h0, w0), (ratio, pad)


def predict_image(model, device, img_bgr, conf_threshold=None, iou_threshold=None, verbose=False):
    img_tensor, (h0, w0), (ratio, pad) = preprocess(img_bgr)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        predictions = model(img_tensor)
        if verbose:
            for i, p in enumerate(predictions):
                obj_conf = p[:, 4::13].sigmoid().view(-1)
                cls_conf = p[:, 5::13].sigmoid().view(-1) if Config.NUM_CLASSES > 0 else torch.tensor(0)
                top_obj = obj_conf.topk(min(5, len(obj_conf)))
                top_cls = cls_conf.topk(min(5, len(cls_conf)))
                print(f"  Layer {i} (stride={Config.STRIDES[i]}) shape={p.shape}: "
                      f"max_obj={obj_conf.max().item():.4f} "
                      f"mean_obj={obj_conf.mean().item():.4f} "
                      f"top5_obj={top_obj.values.tolist()}")
        all_dets = process_predictions(predictions)
        results = _filter_dets(all_dets, img_tensor.shape[0], conf_threshold if conf_threshold is not None else Config.CONF_THRESHOLD,
                               iou_threshold if iou_threshold is not None else Config.IOU_THRESHOLD)
        if verbose:
            r = results[0]
            print(f"  After NMS: {len(r['boxes'])} boxes")
            for i in range(len(r["boxes"])):
                print(f"    [{Config.CLASS_NAMES[r['class_ids'][i]]}] "
                      f"score={r['scores'][i]:.3f} box={r['boxes'][i].tolist()}")

    return results[0], (h0, w0), (ratio, pad)


def draw_boxes(img_bgr, detections, ratio=1.0, pad=(0, 0)):
    for box, score, class_id in zip(
        detections["boxes"], detections["scores"], detections["class_ids"]
    ):
        x1, y1, x2, y2 = box
        dw, dh = pad
        x1 = int((x1 - dw) / ratio)
        y1 = int((y1 - dh) / ratio)
        x2 = int((x2 - dw) / ratio)
        y2 = int((y2 - dh) / ratio)

        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{Config.CLASS_NAMES[class_id]} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_bgr, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_bgr


def predict_single(image_path, output_path=None, model=None, device=None):
    if model is None:
        model, device = load_model()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detections, (h0, w0), (ratio, pad) = predict_image(model, device, img_bgr)
    img_bgr = draw_boxes(img_bgr, detections, ratio, pad)

    if output_path is None:
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(Config.PLOTS_DIR, f"{name}_det{ext}")

    cv2.imwrite(output_path, img_bgr)
    print(f"Saved to {output_path}")

    for box, score, class_id in zip(
        detections["boxes"], detections["scores"], detections["class_ids"]
    ):
        print(f"  {Config.CLASS_NAMES[class_id]:>15s}: {score:.3f}  @ {box.tolist()}")

    return detections


def predict_directory(input_dir, output_dir=None, model=None, device=None,
                      conf_threshold=None, iou_threshold=None):
    if model is None:
        model, device = load_model()
    if output_dir is None:
        output_dir = os.path.join(Config.PLOTS_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(exts)
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")
    for i, filename in enumerate(image_files):
        img_bgr = cv2.imread(os.path.join(input_dir, filename))
        if img_bgr is None:
            print(f"  [SKIP] {filename}")
            continue

        detections, (h0, w0), (ratio, pad) = predict_image(
            model, device, img_bgr, conf_threshold, iou_threshold
        )
        img_bgr = draw_boxes(img_bgr, detections, ratio, pad)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_bgr)

        n = len(detections["boxes"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(image_files)}] {filename}: {n} objects")

    print(f"Done. Results saved to {output_dir}")


def main():
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best.pt")
    last_path = os.path.join(Config.CHECKPOINT_DIR, "last.pt")
    checkpoint = best_path if os.path.exists(best_path) else last_path
    if not os.path.exists(checkpoint):
        print(f"No checkpoint found ({best_path} or {last_path}). Train the model first.")
        return

    model, device = load_model(checkpoint)

    # 先单张跑一次调试看看模型输出
    import numpy as np
    img_debug = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    print("DEBUG: raw model output on random 416x416 image:")
    predict_image(model, device, img_debug, verbose=True, conf_threshold=0.01)

    test_dir = os.path.join(Config.DATA_DIR, "images", "test")
    if os.path.isdir(test_dir):
        test_files = sorted(
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if test_files:
            print(f"Running inference on test set ({len(test_files)} images)...")
            out_dir = os.path.join(Config.PLOTS_DIR, "test_predictions")
            predict_directory(test_dir, out_dir, model, device)
            return

    val_dir = os.path.join(Config.DATA_DIR, "images", "val")
    if os.path.isdir(val_dir):
        val_files = sorted(
            f for f in os.listdir(val_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if val_files:
            sample = random.sample(val_files, min(10, len(val_files)))
            print(f"Running inference on {len(sample)} random val images...")
            out_dir = os.path.join(Config.PLOTS_DIR, "sample_predictions")
            os.makedirs(out_dir, exist_ok=True)
            for f in sample:
                img_bgr = cv2.imread(os.path.join(val_dir, f))
                if img_bgr is None:
                    continue
                detections, (h0, w0), (ratio, pad) = predict_image(model, device, img_bgr)
                img_bgr = draw_boxes(img_bgr, detections, ratio, pad)
                cv2.imwrite(os.path.join(out_dir, f), img_bgr)
                n = len(detections["boxes"])
                print(f"  {f}: {n} objects")
                for box, score, class_id in zip(
                    detections["boxes"], detections["scores"], detections["class_ids"]
                ):
                    print(f"    {Config.CLASS_NAMES[class_id]:>15s}: {score:.3f}")
            print(f"Done. Results saved to {out_dir}")
            return

    print("No images found for inference.")


if __name__ == "__main__":
    main()
