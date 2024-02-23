import numpy as np
import yaml
import paddle
import paddle.nn.functional as F


# 解析yaml配置文件
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding="utf8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]

        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]

        self.category_num = data["MODEL"]["NC"]

        print("Load yaml sucess...")


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.trainable:
                self.shadow[name] = param.numpy().copy()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.trainable:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.numpy() + self.decay * self.shadow[name]
                self.shadow[name] = new_average.copy()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.trainable:
                assert name in self.shadow
                self.backup[name] = param.numpy().copy()
                param.set_value(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.trainable:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}


# 后处理(归一化后的坐标)
def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    total_bboxes, output_bboxes = [], []
    # 将特征图转换为检测框的坐标
    N, C, H, W = preds.shape
    bboxes = paddle.to_tensor(paddle.zeros((N, H, W, 6)))
    pred = preds.transpose((0, 2, 3, 1))
    # 前背景分类分支
    pobj = paddle.unsqueeze(pred[:, :, :, 0], axis=-1)
    # 检测框回归分支
    preg = pred[:, :, :, 1:5]
    # 目标类别分类分支
    pcls = pred[:, :, :, 5:]
    # 检测框置信度
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(axis=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(axis=-1).astype("float32")

    # 检测框的坐标
    gy, gx = paddle.meshgrid([paddle.arange(H), paddle.arange(W)])
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
    bcx = (preg[..., 0].tanh() + gx) / W
    bcy = (preg[..., 1].tanh() + gy) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape([N, (H * W), 6])
    total_bboxes.append(bboxes)

    batch_bboxes = paddle.concat(total_bboxes, axis=1)

    # 对检测框进行NMS处理
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append(obj_score)
            c.append(category)
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # PaddlePaddle NMS
        if len(b) > 0:
            # print(np.array(b).shape, np.array(c).shape, np.array(s).shape)
            b = paddle.to_tensor(b).squeeze(-1)
            c = paddle.to_tensor(c).reshape([-1])
            s = paddle.to_tensor(s, dtype="float32").squeeze(-1)
            keep = paddle.vision.ops.nms(
                boxes=b,
                scores=s,
                category_idxs=c,
                categories=paddle.arange(0, 10),
                iou_threshold=nms_thresh,
            )
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(paddle.to_tensor(output).squeeze(-1))
    return output_bboxes
