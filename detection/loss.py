import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def plain_iou(box1, box2, xywh=True, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
    return inter / union


class YOLOLoss(nn.Module):
    def __init__(self, nc=8, anchors=None, strides=None, box_gain=7.5, cls_gain=0.5,
                 obj_gain=0.7, dfl_gain=0.0):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.na = 3
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.strides = strides if strides is not None else [8, 16, 32]
        self.nl = len(self.strides)

        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))

    def _make_grids(self, shape, device, dtype, layer_idx):
        _, _, ny, nx = shape[0], shape[1], shape[2], shape[3]
        yv, xv = torch.meshgrid(
            torch.arange(ny, device=device, dtype=dtype),
            torch.arange(nx, device=device, dtype=dtype),
            indexing='ij'
        )
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2))
        anchor_grid = (self.anchors[layer_idx].clone().to(device, dtype).view(1, self.na, 1, 1, 2)
                       .expand(1, self.na, ny, nx, 2))
        return grid, anchor_grid

    def _build_targets(self, p, targets, img_size):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        targets_all = targets.repeat(na, 1, 1)
        targets_all = torch.cat((targets_all, ai[..., None]), 2)

        for i in range(self.nl):
            anchors_per_layer = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets_all * gain
            if nt:
                r = t[:, :, 4:6] / anchors_per_layer[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < 4.0
                t = t[j]
            else:
                t = targets_all[0]

            bc = t[:, 0].long()
            a = t[:, 6].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            gi, gj = gij.T

            indices.append((bc, a, gj.clamp_(0, (gain[3] - 1).long()), gi.clamp_(0, (gain[2] - 1).long())))
            tbox.append(torch.cat((gxy - gij.float(), gwh), 1))
            anch.append(anchors_per_layer[a])
            tcls.append(t[:, 1].long())

        return tcls, tbox, indices, anch

    def forward(self, p, targets, img_size):
        device = p[0].device
        dtype = p[0].dtype
        batch_size = p[0].shape[0]

        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors_target = self._build_targets(p, targets, img_size)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            bs, _, h, w = pi.shape
            pi = pi.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            tobj = torch.zeros((bs, self.na, h, w), device=device, dtype=dtype)

            n_pos = b.shape[0]
            if n_pos:
                pi_subset = pi[b, a, gj, gi]
                if pi_subset.ndim == 1:
                    pi_subset = pi_subset.unsqueeze(0)

                grid, anchor_grid = self._make_grids(pi.shape, device, dtype, i)
                grid_xy = grid[0, a, gj, gi]
                anchor_grid_wh = anchor_grid[0, a, gj, gi]
                if grid_xy.ndim == 1:
                    grid_xy = grid_xy.unsqueeze(0)
                if anchor_grid_wh.ndim == 1:
                    anchor_grid_wh = anchor_grid_wh.unsqueeze(0)

                pxy = pi_subset[:, :2].sigmoid() * 2 - 0.5
                pwh = (pi_subset[:, 2:4].sigmoid() * 2) ** 2 * anchor_grid_wh
                pbox = torch.cat((pxy + grid_xy, pwh), 1)

                tbox_i = tbox[i]
                if tbox_i.ndim == 1:
                    tbox_i = tbox_i.unsqueeze(0)

                gij = torch.stack([gi, gj], dim=1).float().to(device)
                tbox_pixel = torch.cat((tbox_i[:, :2] + gij, tbox_i[:, 2:4]), 1)

                ciou = bbox_iou(pbox, tbox_pixel, xywh=True).detach()
                box_loss += (1.0 - ciou).mean()

                with torch.no_grad():
                    obj_target = plain_iou(pbox, tbox_pixel, xywh=True)
                tobj[b, a, gj, gi] = obj_target.squeeze().type(tobj.dtype)

                tcls_mask = torch.zeros_like(pi_subset[:, 5:])
                tcls_mask[torch.arange(n_pos), tcls[i]] = 1.0
                cls_loss += self.bce(pi_subset[:, 5:], tcls_mask).sum(1).mean()

            obji = self.bce(pi[..., 4], tobj)
            pos_mask = tobj > 0
            obj_loss_pos = obji[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=device)
            obj_loss_neg = obji[~pos_mask].mean()
            obj_loss += obj_loss_pos + self.obj_gain * obj_loss_neg

        box_loss *= self.box_gain
        cls_loss *= self.cls_gain
        obj_loss *= 1.0

        total_loss = box_loss + cls_loss + obj_loss
        return total_loss, (box_loss, cls_loss, obj_loss)
