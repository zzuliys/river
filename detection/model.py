import torch
import torch.nn as nn
import math


def autopad(k, p=None, d=1):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    def __init__(self, nc=8, anchors=(), ch=(), stride=()):
        super().__init__()
        self.nc = nc
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.anchors = anchors
        self.stride = stride
        self.ch = ch
        self.no = nc + 5

        c2 = (self.no) * self.na
        self.m = nn.ModuleList(nn.Conv2d(x, c2, 1) for x in ch)

        self._initialize_biases()

    def _initialize_biases(self):
        for mi, s in zip(self.m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999))

    def forward(self, x):
        z = []
        for i in range(self.nl):
            z.append(self.m[i](x[i]))
        return z


class YOLONet(nn.Module):
    def __init__(self, nc=8, anchors=None, ch=(3,)):
        super().__init__()
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        self.nc = nc

        # --- backbone ---
        self.stem = Conv(3, 32, 3, 2)
        self.stage1 = C3(32, 64, n=1)
        self.stage1_down = Conv(64, 128, 3, 2)
        self.stage2 = C3(128, 128, n=2)
        self.stage2_down = Conv(128, 256, 3, 2)
        self.stage3 = C3(256, 256, n=3)
        self.stage3_down = Conv(256, 512, 3, 2)
        self.stage4 = C3(512, 512, n=1)
        self.sppf = SPPF(512, 256)

        # --- neck (FPN + PAN) ---
        self.p5_down = Conv(256, 512, 3, 2)
        self.neck_cv1 = Conv(512, 256, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_c3_1 = C3(768, 256, n=1, shortcut=False)

        self.neck_cv2 = Conv(256, 128, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_c3_2 = C3(384, 128, n=1, shortcut=False)

        self.neck_down1 = Conv(128, 128, 3, 2)
        self.neck_c3_3 = C3(384, 256, n=1, shortcut=False)
        self.neck_down2 = Conv(256, 256, 3, 2)
        self.neck_c3_4 = C3(768, 512, n=1, shortcut=False)

        # --- head ---
        stride = [8, 16, 32]
        ch_head = [128, 256, 512]
        self.detect = Detect(nc=self.nc, anchors=anchors, ch=ch_head, stride=stride)

    def forward(self, x):
        # backbone
        x0 = self.stem(x)                         # 208
        x1 = self.stage1(x0)                       # 208
        x2 = self.stage1_down(x1)                  # 104
        x2 = self.stage2(x2)                       # 104
        x3 = self.stage2_down(x2)                  # 52
        p3 = self.stage3(x3)                       # 52  (P3)
        x4 = self.stage3_down(p3)                  # 26
        p4 = self.stage4(x4)                       # 26  (P4)
        x5 = self.sppf(p4)                         # 26
        p5 = self.p5_down(x5)                       # 13  (P5)

        # neck
        n4_up = self.upsample1(self.neck_cv1(p5))          # 26
        n4 = self.neck_c3_1(torch.cat([n4_up, p4], dim=1)) # 26
        n3_up = self.upsample2(self.neck_cv2(n4))           # 52
        n3 = self.neck_c3_2(torch.cat([n3_up, p3], dim=1)) # 52 (N3)

        n4_d = self.neck_c3_3(torch.cat([self.neck_down1(n3), n4], dim=1))  # 26 (N4)
        n5_d = self.neck_c3_4(torch.cat([self.neck_down2(n4_d), p5], dim=1)) # 13 (N5)

        return self.detect([n3, n4_d, n5_d])

    def get_anchors_and_stride(self):
        anchors = []
        strides = []
        for a, s in zip(self.detect.anchors, self.detect.stride):
            anchors.append(torch.tensor(a).float().view(-1, 2))
            strides.append(torch.tensor(s).float())
        return anchors, torch.tensor(strides).float()
