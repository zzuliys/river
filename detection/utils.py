import torch
import numpy as np
from config import Config


def nms(boxes, scores, iou_threshold=None):
    if iou_threshold is None:
        iou_threshold = Config.IOU_THRESHOLD
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = (iou <= iou_threshold).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        order = order[idx + 1]

    return torch.tensor(keep)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def process_predictions(predictions, conf_threshold=None):
    if conf_threshold is None:
        conf_threshold = Config.CONF_THRESHOLD
    device = predictions[0].device
    dtype = predictions[0].dtype
    bsz = predictions[0].shape[0]
    nc = Config.NUM_CLASSES
    anchors_cfg = [torch.tensor(a).float().to(device).view(-1, 2) for a in Config.ANCHORS]

    all_dets = []
    for layer_idx, pred in enumerate(predictions):
        _, _, h, w = pred.shape
        pred = pred.view(bsz, 3, 5 + nc, h, w).permute(0, 1, 3, 4, 2).contiguous()

        stride = Config.STRIDES[layer_idx]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij'
        )
        anchors = anchors_cfg[layer_idx].view(1, 3, 1, 1, 2)

        box_xy = (pred[..., 0:2].sigmoid() * 2 - 0.5
                  + torch.stack([grid_x, grid_y], dim=-1)) * stride
        box_wh = (pred[..., 2:4].sigmoid() * 2) ** 2 * anchors * stride
        box = torch.cat([box_xy, box_wh], dim=-1)
        obj = pred[..., 4:5].sigmoid()
        cls = pred[..., 5:].sigmoid()

        all_dets.append(torch.cat([box, obj, cls], dim=-1).view(bsz, -1, 5 + nc))

    return torch.cat(all_dets, dim=1)


def _filter_dets(all_dets, bsz, conf_threshold, iou_threshold):
    results = []
    for b in range(bsz):
        dets = all_dets[b]
        obj_conf = dets[:, 4]
        cls_conf, cls_id = dets[:, 5:].max(dim=1)
        score = obj_conf * cls_conf

        mask = score > conf_threshold
        dets = dets[mask]
        score = score[mask]
        cls_id = cls_id[mask]

        if len(dets) == 0:
            results.append({
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "class_ids": np.zeros(0, dtype=int),
            })
            continue

        boxes = xywh2xyxy(dets[:, :4])
        keep = nms(boxes, score, iou_threshold)

        if len(keep) == 0:
            results.append({
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "class_ids": np.zeros(0, dtype=int),
            })
            continue

        results.append({
            "boxes": boxes[keep].cpu().numpy(),
            "scores": score[keep].cpu().numpy(),
            "class_ids": cls_id[keep].cpu().numpy().astype(int),
        })
    return results


def decode_detections(model, img, conf_threshold=None, iou_threshold=None, device=None):
    if conf_threshold is None:
        conf_threshold = Config.CONF_THRESHOLD
    if iou_threshold is None:
        iou_threshold = Config.IOU_THRESHOLD
    if device is None:
        device = img.device

    predictions = model(img)
    all_dets = process_predictions(predictions)
    return _filter_dets(all_dets, img.shape[0], conf_threshold, iou_threshold)
