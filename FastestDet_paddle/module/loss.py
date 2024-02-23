import math
import paddle
import paddle.nn as nn


class DetectorLoss(nn.Layer):
    def __init__(self, device):
        super(DetectorLoss, self).__init__()
        self.device = device

    def bbox_iou(self, box1, box2, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (paddle.minimum(b1_x2, b2_x2) - paddle.maximum(b1_x1, b2_x1)).clip(0) * \
        (paddle.minimum(b1_y2, b2_y2) - paddle.maximum(b1_y1, b2_y1)).clip(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = paddle.fmax(b1_x2, b2_x2) - paddle.fmin(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = paddle.fmax(b1_y2, b2_y2) - paddle.fmin(b1_y1, b2_y1)  # convex height

        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = paddle.pow(s_cw**2 + s_ch**2, 0.5)
        sin_alpha_1 = paddle.abs(s_cw) / sigma
        sin_alpha_2 = paddle.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = paddle.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = paddle.cos(paddle.asin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - paddle.exp(gamma * rho_x) - paddle.exp(gamma * rho_y)
        omiga_w = paddle.abs(w1 - w2) / paddle.fmax(w1, w2)
        omiga_h = paddle.abs(h1 - h2) / paddle.fmax(h1, h2)
        shape_cost = paddle.pow(1 - paddle.exp(-1 * omiga_w), 4) + paddle.pow(
            1 - paddle.exp(-1 * omiga_h), 4
        )
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou

    def build_target(self, preds, targets):
        N, C, H, W = preds.shape
        # batch存在标注的数据
        gt_box, gt_cls, ps_index = [], [], []
        # 每个网格的四个顶点为box中心点会归的基准点
        quadrant = paddle.to_tensor(
            [[0, 0], [1, 0], [0, 1], [1, 1]], dtype="float32", place=self.device
        )

        if targets.shape[0] > 0:
            # 将坐标映射到特征图尺度上
            scale = paddle.to_tensor(paddle.ones(6, dtype="float32"), place=self.device)
            scale[2:] = paddle.to_tensor(preds.shape, dtype="float32")[[3, 2, 3, 2]]
            gt = targets * scale
            
            # 扩展维度复制数据
            gt = paddle.tile(gt, [4, 1, 1])

            # 过滤越界坐标
            quadrant = paddle.tile(quadrant, [gt.shape[1], 1, 1]).transpose((1, 0, 2))
            gij = (gt[..., 2:4]).astype("int64") + quadrant
            j = paddle.where(gij < H, gij, 0).min(axis=-1) > 0

            # 前景的位置下标
            gi, gj = gij[j].T
            batch_index = gt[..., 0].astype("int64")[j]
            ps_index.append((batch_index, gi, gj))

            # 前景的box
            gbox = gt[..., 2:][j]
            gt_box.append(gbox)

            # 前景的类别
            gt_cls.append(gt[..., 1].astype("int64")[j])
        return gt_box, gt_cls, ps_index

    def forward(self, preds, targets):
        # 初始化loss值
        cls_loss = paddle.zeros([1])
        iou_loss = paddle.zeros([1])
        obj_loss = paddle.zeros([1])
        # 定义obj和cls的损失函数
        BCEcls = nn.NLLLoss()
        # smmoth L1相比于bce效果最好
        BCEobj = nn.SmoothL1Loss(reduction="none")

        # 构建ground truth
        gt_box, gt_cls, ps_index = self.build_target(preds, targets)

        pred = preds.transpose((0, 2, 3, 1))
        # 前背景分类分支
        pobj = pred[:, :, :, 0]
        # 检测框回归分支
        preg = pred[:, :, :, 1:5]
        # 目标类别分类分支
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = paddle.zeros_like(pobj)
        factor = paddle.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            # 计算检测框回归loss
            b, gx, gy = ps_index[0]
            ptbox = paddle.ones(preg[b, gy, gx].shape)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H

            # 计算检测框IOU loss
            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter

            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            # 计算iou loss
            iou = iou[f]
            iou_loss = (1.0 - iou).mean()

            # 计算目标类别分类分支loss
            ps = paddle.log(pcls[b, gy, gx])
            cls_loss = BCEcls(ps, gt_cls[0][f])

            # iou aware
            tobj[b, gy, gx] = iou.astype("float32")
            # 统计每个图片正样本的数量
            n = paddle.bincount(b)
            factor[b, gy, gx] = (1.0 / (n[b] / (H * W))) * 0.25

        # 计算前背景分类分支loss
        obj_loss = (BCEobj(pobj, tobj) * factor).mean()

        # 计算总loss
        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss

        return iou_loss, obj_loss, cls_loss, loss
