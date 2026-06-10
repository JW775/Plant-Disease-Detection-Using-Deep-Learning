from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from model import ResNet18 # 导入模型
# from model import Residual

# 导入官方的 ResNet18 模型
from torchvision import models

"""
    Team Member: JiaYi Wu
    Description: Model Training,
    
    Data Loading & Augmentation:Loads the local plant dataset and applies various data_new augmentations (e.g., random crop, flip, color jitter) 
    to the training set to prevent overfitting, while applying only basic resizing to the validation set.
    
    Model Setup & Transfer Learning: Loads the official ImageNet-pretrained ResNet18 model. Employs a freezing strategy: freezes the backbone network, 
    unfreezes only Layer4 and the FC layer, and replaces the FC layer with a custom number of classes for fine-tuning.
    
    Training Pipeline & Strategy: Trains for 30 epochs using the Adam optimizer and a Cosine Annealing learning rate scheduler. After each epoch,
     it calculates and logs the loss and accuracy for both sets, and automatically saves the model weights that achieve the highest validation accuracy.
    
    Result Visualization: After training, uses Matplotlib to plot the line charts of Loss and Accuracy over epochs, providing an intuitive view of the model's convergence.
"""


# ---加载数据集---

# 处理训练集和验证集数据
# Data Loading & Augmentation
def train_val_data_process():
    data_dir = 'D:/develop/pythonProjects/plant/dataset'
    # 数据预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 训练集预处理： 数据增强（反转，旋转） 防止过拟合
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 调整大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(20),      # 随机旋转

        # 新增的数据增强
        # 颜色抖动
        # 亮度， 对比度， 饱和度， 色相
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # 随即灰度
        transforms.RandomGrayscale(p=0.1),
        # 高斯模型
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1,2.0)),
        transforms.ToTensor(),              # 转为张量
        normalize,

        # 随机擦除
        transforms.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value=0, inplace=False)
    ])

    # 验证集数据预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 记载本地文件夹
    train_data = ImageFolder(root=data_dir+ '/train', transform=train_transform)
    val_data = ImageFolder(root=data_dir+ '/valid', transform=val_transform)

    print(f"训练集样本数：{len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"类别数量: {len(train_data.classes)}")

    # 设置Batch
    train_dataloader = data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    val_dataLoader = data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=2)

    return train_dataloader, val_dataLoader, len(train_data.classes)

# ----------

# ---模型训练---    # num_epoches - 训练轮次
def train_model_process(model, train_dataloader, val_dataloader, num_epoches):
    # 初始化
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义Adam优化器, 学习率为0.0005 (利于训练模型的时候更新参数) -梯度下降法 -根据梯度修改权重
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 告诉优化器只去封信最新的模型就好了
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 每10轮变成原来的0.1
    # gamma = 0.1 表示变成原来的0.1倍
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epoches,
        eta_min=1e-6 # 最小学习率
    )

    # 定义损失函数 - 计算Loss
    criterion = nn.CrossEntropyLoss() # 交叉熵损失(分类常用)

    # 把模型放到训练设备中
    model = model.to(device)

    # 复制当前模型的参数
    # 深拷贝， 独立记录当前最好的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确值
    best_acc = 0.0

    # 训练集损失值列表
    train_losses = []
    # 验证集损失值列表
    val_losses = []
    # 训练集准确度列表
    train_accs = []
    # 验证集准确度列表
    val_accs = []
    # 当前时间
    since = time.time()

    # 训练轮次
    for epoch in range(num_epoches):
        print(f"Epoch {epoch+1} / {num_epoches}")
        print("-" * 20)

        # 初始化参数
        train_loss = 0.0
        train_acc = 0.0

        val_loss = 0.0
        val_acc = 0.0

        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 把特征放入训练设备中
            b_x = b_x.to(device)
            # 把标签放入训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程， 输入为一个batch， 输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向梯度计算
            loss.backward()
            # 根据反向传播的梯度信息来更新网路参数， 以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确， 则准确度 train_acc + 1
            train_acc += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        # 验证过程
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入验证设备中
            b_x = b_x.to(device)
            # 将标签放入验证设备中
            b_y = b_y.to(device)
            # 将模型设置为评估模式
            model.eval()
            # 前向传播过程， 输入为一个batch， 输出为一个batch中对应的预测
            output = model(b_x)

            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加 # loss.item() --> 平均loss值
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确， 验证集准确度+1
            val_acc += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_losses.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_accs.append(train_acc.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_losses.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_accs.append(val_acc.double().item() / val_num)

        # 打印
        print(f"{epoch+1} epoch- Train Loss: {train_losses[-1]:.4f} Train Acc: {train_accs[-1]:.4f}")
        print(f"{epoch+1} epoch- Val Loss: {val_losses[-1]:.4f} Val Acc: {val_accs[-1]:.4f}")

        if val_accs[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_accs[-1]
            # 保存当前的参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 更新学习率
        scheduler.step()

        # 训练耗费时间
        time_use = time.time() - since
        print(f"训练耗费时间: {(time_use//60):.0f}m {(time_use%60):.0f}s")
        print("-" * 20)

    # 选择最优参数
    # 加载最高准确度下的模型参数
    torch.save(best_model_wts, "D:/develop/pythonProjects/Plant/best_model_process.pth")



    train_process = pd.DataFrame(data={"epoch": range(num_epoches),
                                            "train_losses": train_losses,
                                           "val_losses": val_losses,
                                           "train_accs": train_accs,
                                           "val_accs": val_accs})


    return train_process


# 画图
# Result Visualization
def matplot_acc_loss(train_process):
    # 创建画布
    plt.figure(figsize=(12,4))
    # 1行2列的第一张图 -Loss值
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_losses, 'ro-', label="Train Loss") # 红色圆形实线
    plt.plot(train_process["epoch"], train_process.val_losses, 'bs-', label="Val Loss") # 蓝色方形实现
    plt.legend() # 显示图例， 显示线代表什么
    # 给坐标轴贴上标签
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 1行2列的第二张图 -准确值
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_accs, 'ro-', label="Train Acc")
    plt.plot(train_process["epoch"], train_process.val_accs, 'bs-', label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()


# 主方法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataloader, val_dataloader, num_classes = train_val_data_process()

    # 加载官方的ImageNet 与训练的 ResNet 18
    print("正在加载 ImageNet 预训练的 ResNet18")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 冻结特征提取层和Layer 4（保留常规通用的视觉经验，加速初期收敛）
    # 遍历 1: 全局冻结
    for param in model.parameters():
        param.requires_grad = False

    # 遍历 2: 解冻Layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 遍历 3: 解冻FC层
    for param in model.fc.parameters():
        param.requires_grad = True
    # 替换最后一层分类头（44分类 + Non-Plant）
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 把模型放到设备（GPU）
    model = model.to(device)

    # 开始训练
    train_process = train_model_process(model, train_dataloader, val_dataloader, 30)
    matplot_acc_loss(train_process)
    # ======================================================================
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"当前使用设备 {device}")
    #
    #
    # train_dataloader, val_dataloader, num_classes =train_val_data_process()
    #
    #
    # # # 加载训练过权重
    # # try:
    # #     checkpoint = torch.load("D:/develop/Plant/best_model.pth", map_location=device)
    # #     model.load_state_dict(checkpoint)
    # #     print("加载预训练权重，继续训练")
    # # except Exception as e:
    # #     print(f"未加载成功： {e}")
    #
    # model = ResNet18(num_classes=num_classes).to(device)
    # train_process = train_model_process(model, train_dataloader, val_dataloader, 30)
    # matplot_acc_loss(train_process)
