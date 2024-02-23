import os
import math
import paddle
import paddle.optimizer as optimizer
import argparse
from tqdm import tqdm
from paddle.io import DataLoader


from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.loss import DetectorLoss
from module.detector import Detector

from torch.utils.tensorboard import SummaryWriter

# 指定后端设备CUDA&CPU
device = paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
print(device)


# python3 train.py --yaml ./configs/coco.yaml --weight ./checkpoint/weight_AP05:0.000000_100-epoch.pth
class FastestDet:
    def __init__(self):
        # 指定训练配置文件
        parser = argparse.ArgumentParser()
        parser.add_argument("--yaml", type=str, default="", help=".yaml config")
        parser.add_argument("--weight", type=str, default=None, help=".weight config")

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"

        # 解析yaml配置文件
        self.cfg = LoadYaml(opt.yaml)
        # 初始化模型结构
        if opt.weight is not None:
            print("load weight from:%s" % opt.weight)
            self.model = Detector(self.cfg.category_num, True).to(device)
            self.model.set_state_dict(paddle.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False).to(device)

        # # 打印网络各层的张量维度
        # paddle.summary(
        #     self.model, input_size=(1, 3, self.cfg.input_height, self.cfg.input_width)
        # )

        # 构建优化器
        print("use SGD optimizer")
        self.optimizer = optimizer.Momentum(
            parameters=self.model.parameters(),
            learning_rate=self.cfg.learn_rate,
            momentum=0.949,
            weight_decay=0.0005,
        )
        # 学习率衰减策略
        self.scheduler = optimizer.lr.MultiStepDecay(
            learning_rate=self.cfg.learn_rate, milestones=self.cfg.milestones, gamma=0.1
        )

        # 定义损失函数
        self.loss_function = DetectorLoss(device)

        # 定义验证函数
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, device)

        # 数据集加载
        val_dataset = TensorDataset(
            self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False
        )
        train_dataset = TensorDataset(
            self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, False
        )

        # 验证集

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=False,
            persistent_workers=True,
        )
        # 训练集
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
        )

    def train(self):
        logger = SummaryWriter(log_dir="./log")
        # 迭代训练
        batch_num = 0
        print("Starting training for %g epochs..." % self.cfg.end_epoch)
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                # 数据预处理
                imgs = paddle.to_tensor(imgs, dtype="float32", place=device) / 255.0
                targets = paddle.to_tensor(targets)
                # print(targets[0])
                # 模型推理
                preds = self.model(imgs)
                # loss计算
                iou, obj, cls, total = self.loss_function(preds, targets)

                # 写入日志信息
                logger.add_scalar("total_loss", np.array(total), epoch + 1)
                logger.add_scalar("cls_loss", np.array(cls), epoch + 1)
                logger.add_scalar("obj_loss", np.array(obj), epoch + 1)
                logger.add_scalar("iou_loss", np.array(iou), epoch + 1)

                # 反向传播求解梯度
                total.backward()
                # 更新模型参数
                self.optimizer.step()
                self.optimizer.clear_grad()
                # 学习率预热
                for k_i in range(1):
                    g = self.optimizer._default_dict
                    warmup_num = 5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = paddle.to_tensor(batch_num / warmup_num) ** 4
                        g["lr"] = self.cfg.learn_rate * scale
                    lr = g["lr"]
                # 打印相关训练信息
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                    epoch,
                    lr,
                    iou,
                    obj,
                    cls,
                    total,
                )
                pbar.set_description(info)
                batch_num += 1

            # 模型验证及保存
            if epoch % 10 == 0 and epoch > 0:
                # 模型评估
                self.model.eval()
                print("computer mAP...")
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)

                logger.add_scalar("mAP", np.array(mAP05), epoch + 1)
                paddle.save(
                    self.model.state_dict(),
                    "checkpoint/weight_AP05:%f_%d-epoch.pdparams" % (mAP05, epoch),
                )

            # 学习率调整
            self.scheduler.step()


if __name__ == "__main__":
    model = FastestDet()
    model.train()
