import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# ==================== 绝对路径配置 ====================
DATA_PATH = r'C:\trian\trian' # 确保这里是47个类别的文件夹
OUTPUT_DIR = r'C:\trian\model\output_47class'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'plant_model_47class.pth')
TRAIN_CURVE_PATH = os.path.join(OUTPUT_DIR, 'training_curve.png')
CONF_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 针对8GB显存的优化配置 ====================
BATCH_SIZE = 16 # 批大小
GRADIENT_ACCUMULATION = 2 # 梯度累积，模拟 batch_size=32
USE_AMP = True # 混合精度训练
EPOCHS = 20 # 训练轮次
LEARNING_RATE = 0.0001
# ============================================================

# -------------------- 1. 数据预处理 --------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),                    # 先放大一点点
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 🔥 关键：随机裁剪，模拟叶子只占画面一部分
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),             # 有时照片会上下颠倒
    transforms.RandomRotation(30),                    # 旋转角度更大
    transforms.ColorJitter(
        brightness=0.4,   # 亮度变化剧烈（模拟过曝/欠曝）
        contrast=0.4,     # 对比度变化
        saturation=0.4,   # 饱和度变化
        hue=0.2           # 色调变化
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 2. 加载数据 --------------------
print("📂 正在读取数据...")
full_dataset = datasets.ImageFolder(DATA_PATH, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 验证集不需要数据增强
val_dataset.dataset.transform = val_transform

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"✅ 检测到 {num_classes} 个类别")
print(f"类别列表: {class_names}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------- 3. 构建模型 (改用 ResNet18) --------------------
print("🧠 正在加载预训练模型 ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结所有参数，只训练最后一部分以节省显存
for param in model.parameters():
    param.requires_grad = False

# 解冻 layer4 和全连接层（你也可以根据自己的需要解冻更多层）
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# 替换最后的全连接层以匹配类别数
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# -------------------- 4. 训练配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"💻 使用设备: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# 混合精度梯度缩放器
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# -------------------- 5. 训练循环 --------------------
print(f"🚀 开始训练 (Batch Size: {BATCH_SIZE}, 梯度累积: {GRADIENT_ACCUMULATION}, AMP: {USE_AMP})...")
train_losses = []
val_accuracies = []

for epoch in range(1, EPOCHS + 1):
    # ---- 训练 ----
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels) / GRADIENT_ACCUMULATION

        scaler.scale(loss).backward()

        if (i + 1) % GRADIENT_ACCUMULATION == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * GRADIENT_ACCUMULATION

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ---- 验证 ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch:2d}/{EPOCHS}] Loss: {avg_loss:.4f} Val Accuracy: {val_acc:.2f}%")

print("🎉 训练完成！")

# -------------------- 6. 分类报告和混淆矩阵 --------------------
print("\n📊 正在生成分类报告...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

print("\n" + "=" * 50)
print("分类报告")
print("=" * 50)
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# 混淆矩阵
plt.figure(figsize=(20, 18))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (47 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=150)
print(f"✅ 混淆矩阵已保存到 {CONF_MATRIX_PATH}")

# -------------------- 7. 保存模型 --------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 模型已保存到 {MODEL_PATH}")

# -------------------- 8. 画训练曲线 --------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), val_accuracies, marker='s', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Curve')

plt.tight_layout()
plt.savefig(TRAIN_CURVE_PATH, dpi=150)
print(f"📈 训练曲线已保存到 {TRAIN_CURVE_PATH}")