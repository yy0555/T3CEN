import torch
import torch.nn as nn
import lightning.pytorch as L
from torch import optim

import numpy as np

from modules.networks import networks
from torchmetrics import Accuracy

class LitHSGCNN(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        print("Parameters:")
        for k, v in params.items():
            print(f"{k}: {v}")

        self._set_seed()
        self._get_net()
        self._get_loss()
        self.val_acc = Accuracy(task="multiclass", num_classes=self.params["num_classes"])
        # 注册 buffer 以存储最高准确率，支持 DDP 同步和模型恢复
        self.register_buffer("max_val_acc", torch.tensor(0.0))

    def _set_seed(self):
        if self.params["manual_seed"] != -1:
            torch.manual_seed(self.params["manual_seed"])
            np.random.seed(self.params["manual_seed"])
    
    def _get_net(self):
        network_name = self.params["network_name"]
        num_classes = self.params["num_classes"]
        n_groups_hue = self.params["n_groups_hue"]
        n_groups_saturation = self.params["n_groups_saturation"]
        n_groups_luminance = self.params["n_groups_luminance"]
        network_class = networks.parse_name(network_name)
        self.net = network_class(num_classes=num_classes,
                                    n_groups_hue=n_groups_hue,
                                    n_groups_saturation=n_groups_saturation,
                                    n_groups_luminance=n_groups_luminance)
    
    def _get_loss(self):
        loss_name = self.params["loss_name"]
        if loss_name == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss name {loss_name} not recognized.")

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        _, predicted = y_hat.max(1)
        correct = predicted.eq(y).sum().item()
        self.log("train_accuracy", correct / y.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        
        preds = torch.argmax(y_hat, dim=1)
        # 更新 TorchMetrics 内部状态
        self.val_acc.update(preds, y)
        return loss

    def on_validation_epoch_end(self):
        # 计算整个 epoch 的准确率
        curr_val_acc = self.val_acc.compute()
        
        # 更新历史最高值
        if curr_val_acc > self.max_val_acc:
            self.max_val_acc = curr_val_acc
            
        # 记录到日志系统和进度条
        # 注意：这里的 "val_accuracy" 必须与 run.py 中的 monitor_metric 一致
        self.log("val_accuracy", curr_val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("max_acc", self.max_val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 重置以供下一个 epoch 使用
        self.val_acc.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[0], batch[1]
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, sync_dist=True, batch_size=y.size(0))
        _, predicted = y_hat.max(1)
        correct = predicted.eq(y).sum().item()
        self.log("test_accuracy", correct / y.size(0), sync_dist=True, batch_size=y.size(0))

    def configure_optimizers(self):
        lr = self.params["lr"]
        if self.params["optimizer"] == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif self.params["optimizer"] == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.params['optimizer']} not recognized.")
        
        if self.params["scheduler"] == "none":
            return optimizer
        elif self.params["scheduler"] == "cosine":
            T_max = self.params.get("n_epochs", 10)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError(f"Scheduler {self.params['scheduler']} not recognized.")
