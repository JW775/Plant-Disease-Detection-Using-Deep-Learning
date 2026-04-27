import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')   # 强制使用非交互式后端，彻底解决你的保存报错
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------- 数据路径（你的实际情况）--------------------
DATA_PATH = r'C:\trian\trian'

# -------------------- 1. 数据预处理 --------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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

full_dataset = datasets.ImageFolder(DATA_PATH,
                                     transform=train_transform)

# 80% 训练，20% 验证
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 验证集换回不带增强的 transform
val_dataset.dataset.transform = val_transform

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"✅ 检测到 {num_classes} 个类别: {class_names}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------- 3. 构建模型 --------------------
print("🧠 正在加载预训练模型 ResNet18...")
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)

# -------------------- 4. 训练配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

EPOCHS = 10

# -------------------- 5. 训练循环 --------------------
print("🚀 开始训练...")
train_losses = []
val_accuracies = []

for epoch in range(1, EPOCHS + 1):
    # ---- 训练 ----
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ---- 验证 ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
          f"Loss: {avg_loss:.4f}  "
          f"Val Accuracy: {val_acc:.2f}%")

print("🎉 训练完成！")

# -------------------- 6. 分类报告和混淆矩阵 --------------------
print("\n📊 正在生成分类报告...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

print("\n" + "=" * 50)
print("分类报告")
print("=" * 50)
print(classification_report(all_labels, all_preds, target_names=class_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(r'C:\trian\model\output\confusion_matrix.png', dpi=150)
print("\n✅ 混淆矩阵已保存到 output/confusion_matrix.png")

# -------------------- 7. 保存模型 --------------------
os.makedirs(r'C:\trian\model\output', exist_ok=True)
torch.save(model.state_dict(), r'C:\trian\model\output\plant_model.pth')
print("💾 模型已保存到 output/plant_model.pth")

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
plt.savefig(r'C:\trian\model\output\training_curve.png', dpi=150)
print("📈 训练曲线已保存到 output/training_curve.png")
plt.show()