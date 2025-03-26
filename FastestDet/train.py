import os
import math
import torch
import argparse
from tqdm import tqdm
from torch import optim
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.loss import DetectorLoss
from module.detector import Detector

# 指定后端设备CUDA&CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastestDet:
    def __init__(self):
        # 指定训练配置文件
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="configs/apricot.yaml", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='.weight config')
        parser.add_argument('--project', type=str, default='runs/train')
        parser.add_argument('--name', type=str, default='exp')

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"

        # 解析yaml配置文件
        self.cfg = LoadYaml(opt.yaml)
        print(self.cfg)

        # 初始化模型结构
        if opt.weight is not None:
            print("load weight from:%s" % opt.weight)
            self.model = Detector(self.cfg.category_num, True).to(device)
            self.model.load_state_dict(torch.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False).to(device)

        # # 打印网络各层的张量维度
        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

        self.num_workers = 8
        self.batch_size = 32
        self.project = opt.project
        self.name = opt.name
        self.save_dir = self.create_save_directory()
        self.best_map = 0.0  # 初始化最佳mAP

        # 构建优化器
        print("use SGD optimizer")
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.cfg.learn_rate,
                                   momentum=0.949,
                                   weight_decay=0.0005,
                                   )
        # 学习率衰减策略
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                 milestones=self.cfg.milestones,
        #                                                 gamma=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.end_epoch,
            eta_min=self.cfg.learn_rate * 0.1
        )

        # 定义损失函数
        self.loss_function = DetectorLoss(device)

        # 定义验证函数
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, device)

        # 数据集加载
        val_dataset = TensorDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False)
        train_dataset = TensorDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, True)

        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=collate_fn,
                                                            num_workers=self.num_workers,
                                                            drop_last=True,
                                                            persistent_workers=True
                                                            )

        # 验证集
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          collate_fn=collate_fn,
                                                          num_workers=self.num_workers,
                                                          drop_last=False,
                                                          persistent_workers=True
                                                          )

    def create_save_directory(self):
        save_dir_base = os.path.join(self.project, self.name)
        save_dir = save_dir_base
        counter = 1
        while os.path.exists(save_dir):
            save_dir = f"{save_dir_base}{counter}"
            counter += 1

        # 创建目录
        os.makedirs(save_dir, exist_ok=False)
        print(f"模型将保存到：{save_dir}")
        return save_dir

    def train(self):
        # 迭代训练
        batch_num = 0
        print('Starting training for %g epochs...' % self.cfg.end_epoch)
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                # 数据预处理
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
                # 模型推理
                preds = self.model(imgs)

                # loss计算
                iou, obj, cls, total = self.loss_function(preds, targets)
                # 反向传播求解梯度
                total.backward()
                # 更新模型参数
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 学习率预热
                for g in self.optimizer.param_groups:
                    warmup_num = 5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num / warmup_num, 4)
                        g['lr'] = self.cfg.learn_rate * scale
                    lr = g['lr']

                # 打印相关训练信息
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (epoch, lr, iou, obj, cls, total)
                pbar.set_description(info)
                batch_num += 1

            mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)

            # 保存最佳模型
            if mAP05 > self.best_map:
                self.best_map = mAP05
                best_save_path = os.path.join(self.save_dir, "best.pth")
                torch.save(self.model.state_dict(), best_save_path)
                print(f"保存最佳模型，mAP提升至{self.best_map:.6f}")

            # 保存最后一个epoch的模型
            last_save_path = os.path.join(self.save_dir, "last.pth")
            torch.save(self.model.state_dict(), last_save_path)

            # 学习率调整
            self.scheduler.step()


if __name__ == "__main__":
    model = FastestDet()
    model.train()
