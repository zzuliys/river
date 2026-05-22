import os
import time
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from config import Config
from model import YOLONet
from loss import YOLOLoss, xywh2xyxy, bbox_iou
from dataset import YOLODataset
from utils import nms, process_predictions

Config.ensure_dirs()


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def compute_map_per_class(gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes,
                          nc, iou_threshold=0.5):
    ap_per_class = np.zeros(nc)
    for c in range(nc):
        gt_mask = (gt_classes == c)
        pred_mask = (pred_classes == c)

        n_gt = gt_mask.sum()
        if n_gt == 0 and pred_mask.sum() == 0:
            ap_per_class[c] = 0.0
            continue

        if pred_mask.sum() == 0:
            ap_per_class[c] = 0.0
            continue

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        sorted_idx = np.argsort(-pred_scores[pred_mask])
        pred_boxes_c = pred_boxes[pred_mask][sorted_idx]
        gt_boxes_c = gt_boxes[gt_mask]

        if len(gt_boxes_c):
            ious = np.zeros((len(pred_boxes_c), len(gt_boxes_c)))
            for pi in range(len(pred_boxes_c)):
                for gi in range(len(gt_boxes_c)):
                    x1 = max(pred_boxes_c[pi, 0], gt_boxes_c[gi, 0])
                    y1 = max(pred_boxes_c[pi, 1], gt_boxes_c[gi, 1])
                    x2 = min(pred_boxes_c[pi, 2], gt_boxes_c[gi, 2])
                    y2 = min(pred_boxes_c[pi, 3], gt_boxes_c[gi, 3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area_pred = (pred_boxes_c[pi, 2] - pred_boxes_c[pi, 0]) * (pred_boxes_c[pi, 3] - pred_boxes_c[pi, 1])
                    area_gt = (gt_boxes_c[gi, 2] - gt_boxes_c[gi, 0]) * (gt_boxes_c[gi, 3] - gt_boxes_c[gi, 1])
                    ious[pi, gi] = inter / (area_pred + area_gt - inter + 1e-10)

            matched = np.zeros(len(gt_boxes_c), dtype=bool)
            for pi in range(len(pred_boxes_c)):
                best_gi = np.argmax(ious[pi])
                if ious[pi, best_gi] >= iou_threshold and not matched[best_gi]:
                    tp[pi] = 1
                    matched[best_gi] = True
                else:
                    fp[pi] = 1
        else:
            fp[:] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / n_gt if n_gt > 0 else np.zeros_like(tp_cumsum)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        ap_per_class[c] = compute_ap(recall, precision) if n_gt > 0 else 0.0

    return ap_per_class


@torch.no_grad()
def validate(model, val_loader, device, conf_threshold=None, iou_threshold=None):
    if conf_threshold is None:
        conf_threshold = Config.CONF_THRESHOLD
    if iou_threshold is None:
        iou_threshold = Config.IOU_THRESHOLD
    model.eval()
    nc = Config.NUM_CLASSES

    all_gt_boxes = []
    all_gt_classes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []

    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        predictions = model(imgs)

        all_dets = process_predictions(predictions, conf_threshold)

        for b in range(imgs.shape[0]):
            img_targets = targets[targets[:, 0] == b]

            gt_boxes_list = []
            gt_classes_list = []
            for t in img_targets:
                cls_id = int(t[1].item())
                cx, cy, w, h = t[2:6].tolist()
                x1 = (cx - w / 2) * imgs.shape[-1]
                y1 = (cy - h / 2) * imgs.shape[-2]
                x2 = (cx + w / 2) * imgs.shape[-1]
                y2 = (cy + h / 2) * imgs.shape[-2]
                gt_boxes_list.append([x1, y1, x2, y2])
                gt_classes_list.append(cls_id)

            if gt_boxes_list:
                all_gt_boxes.append(np.array(gt_boxes_list))
                all_gt_classes.append(np.array(gt_classes_list))

            dets = all_dets[b]
            obj_conf = dets[:, 4]
            cls_conf, cls_id = dets[:, 5:].max(dim=1)
            score = obj_conf * cls_conf

            mask = score > conf_threshold
            dets = dets[mask]
            score = score[mask]
            cls_id = cls_id[mask]

            if len(dets) == 0:
                continue

            boxes = xywh2xyxy(dets[:, :4])
            nms_idx = nms(boxes, score, iou_threshold)

            if len(nms_idx) == 0:
                continue

            boxes = boxes[nms_idx].cpu().numpy()
            scores_np = score[nms_idx].cpu().numpy()
            cls_np = cls_id[nms_idx].cpu().numpy()

            all_pred_boxes.append(boxes)
            all_pred_scores.append(scores_np)
            all_pred_classes.append(cls_np)

    if len(all_gt_boxes) == 0:
        return 0.0, 0.0

    all_gt_boxes_cat = np.concatenate(all_gt_boxes)
    all_gt_classes_cat = np.concatenate(all_gt_classes)
    all_pred_boxes_cat = np.concatenate(all_pred_boxes) if all_pred_boxes else np.zeros((0, 4))
    all_pred_scores_cat = np.concatenate(all_pred_scores) if all_pred_scores else np.zeros(0)
    all_pred_classes_cat = np.concatenate(all_pred_classes) if all_pred_classes else np.zeros(0)

    ap50 = compute_map_per_class(
        all_gt_boxes_cat, all_gt_classes_cat,
        all_pred_boxes_cat, all_pred_scores_cat, all_pred_classes_cat,
        nc, iou_threshold=Config.IOU_THRESHOLD
    )

    mAP50 = ap50.mean()
    return mAP50, ap50


def train():
    device = torch.device(Config.DEVICE)
    torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
    print(f"Device: {device}")

    img_size = Config.IMAGE_SIZE
    batch_size = Config.BATCH_SIZE

    train_dataset = YOLODataset(Config.DATA_DIR, split='train', img_size=img_size, augment=True)
    val_dataset = YOLODataset(Config.DATA_DIR, split='val', img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        collate_fn=YOLODataset.collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        collate_fn=YOLODataset.collate_fn
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Batch: {batch_size}, Size: {img_size}")

    model = YOLONet(nc=Config.NUM_CLASSES, anchors=Config.ANCHORS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")

    loss_fn = YOLOLoss(
        nc=Config.NUM_CLASSES, anchors=Config.ANCHORS,
        strides=Config.STRIDES,
        box_gain=Config.BOX_GAIN, cls_gain=Config.CLS_GAIN,
        obj_gain=Config.OBJ_GAIN
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )
    scaler = GradScaler() if Config.USE_AMP and device.type == "cuda" else None

    best_mAP = 0.0
    best_epoch = 0
    log_file = os.path.join(Config.LOG_DIR, "train_log.csv")
    header = ("epoch,box_loss,cls_loss,obj_loss,total_loss,"
              "val_box_loss,val_cls_loss,val_obj_loss,val_total_loss,mAP50,time\n")
    with open(log_file, 'w') as f:
        f.write(header)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_box_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            if scaler:
                with autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
                    predictions = model(imgs)
                    total_loss, (box_loss, cls_loss, obj_loss) = loss_fn(
                        predictions, targets, img_size
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(imgs)
                total_loss, (box_loss, cls_loss, obj_loss) = loss_fn(
                    predictions, targets, img_size
                )
                total_loss.backward()
                optimizer.step()

            epoch_box_loss += box_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_obj_loss += obj_loss.item()
            epoch_total_loss += total_loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"box: {box_loss.item():.4f} cls: {cls_loss.item():.4f} "
                      f"obj: {obj_loss.item():.4f} total: {total_loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - t0

        n_batches = len(train_loader)
        avg_box = epoch_box_loss / n_batches
        avg_cls = epoch_cls_loss / n_batches
        avg_obj = epoch_obj_loss / n_batches
        avg_total = epoch_total_loss / n_batches

        val_box_loss = 0.0
        val_cls_loss = 0.0
        val_obj_loss = 0.0
        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                predictions = model(imgs)
                total_loss, (box_loss, cls_loss, obj_loss) = loss_fn(
                    predictions, targets, img_size
                )
                val_box_loss += box_loss.item()
                val_cls_loss += cls_loss.item()
                val_obj_loss += obj_loss.item()
                val_total_loss += total_loss.item()

        n_val = len(val_loader)
        avg_val_box = val_box_loss / n_val
        avg_val_cls = val_cls_loss / n_val
        avg_val_obj = val_obj_loss / n_val
        avg_val_total = val_total_loss / n_val

        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS} | "
              f"Train Loss: {avg_total:.4f} "
              f"(box:{avg_box:.4f} cls:{avg_cls:.4f} obj:{avg_obj:.4f}) | "
              f"Val Loss: {avg_val_total:.4f} "
              f"(box:{avg_val_box:.4f} cls:{avg_val_cls:.4f} obj:{avg_val_obj:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.1f}s")

        mAP50 = 0.0
        if epoch % Config.MAP_EVAL_INTERVAL == 0 or epoch == Config.NUM_EPOCHS:
            print("  Computing mAP...")
            mAP50, ap_per_class = validate(model, val_loader, device)
            print(f"  mAP@50: {mAP50:.4f}")
            for i, ap in enumerate(ap_per_class):
                print(f"    {Config.CLASS_NAMES[i]:>15s}: {ap:.4f}")

        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_box:.6f},{avg_cls:.6f},{avg_obj:.6f},{avg_total:.6f},"
                    f"{avg_val_box:.6f},{avg_val_cls:.6f},{avg_val_obj:.6f},{avg_val_total:.6f},"
                    f"{mAP50:.6f},{epoch_time:.1f}\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mAP50': mAP50,
        }, os.path.join(Config.CHECKPOINT_DIR, "last.pt"))

        if mAP50 >= best_mAP:
            best_mAP = mAP50
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'mAP50': mAP50,
            }, os.path.join(Config.CHECKPOINT_DIR, "best.pt"))
            print(f"  New best mAP@50: {best_mAP:.4f} saved!")

    print(f"\nTraining complete. Best mAP@50: {best_mAP:.4f} at epoch {best_epoch}")

    print("\n--- Test set evaluation ---")
    test_dataset = YOLODataset(Config.DATA_DIR, split='test', img_size=img_size, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        collate_fn=YOLODataset.collate_fn
    )

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  Warning: no checkpoint found, using current model weights")
    model.to(device)
    model.eval()

    mAP50_test, ap_per_class = validate(model, test_loader, device)
    print(f"\nTest Set Results:")
    print(f"  mAP@50: {mAP50_test:.4f}")
    for i, ap in enumerate(ap_per_class):
        print(f"    {Config.CLASS_NAMES[i]:>15s}: {ap:.4f}")


if __name__ == "__main__":
    train()
