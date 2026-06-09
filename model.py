"""
    Team member: JiaYi Wu
    Description: Define the pre-trained model - ResNet18
"""
import torchvision.models as models
import torch.nn as nn

class MyResNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        # 在这里修改最后一层
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)