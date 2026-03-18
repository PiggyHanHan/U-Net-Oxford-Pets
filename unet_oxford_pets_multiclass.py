# -*- coding: utf-8 -*-
"""
U-Net 入门实战：Oxford-IIIT Pet 数据集（多类分割，37类）
数据集特性：
- 图像：宠物照片（RGB）
- 掩码：每个像素为 0（背景）或 1~36（不同宠物品种）
- 使用 torchvision 自动下载，无需手动处理
- 多类分割，适合入门后进阶
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# -------------------- 1. U-Net 模型定义（与之前相同，但注意输出通道数改为37）--------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -------------------- 2. 数据集预处理（恢复多类） --------------------
# 图像变换：缩放到固定大小，转为Tensor
image_size = (256, 256)
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# 掩码变换：也需要缩放（最近邻插值），然后转为LongTensor（保持原始像素值 0~36）
mask_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
])

print("正在下载/加载 Oxford-IIIT Pet 数据集...")
train_dataset = OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types='segmentation',
    download=True,
    transform=image_transform,
    target_transform=mask_transform,
)

# 随机划分训练集和验证集（80%训练，20%验证）
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

print(f"训练集大小：{len(train_dataset)}，验证集大小：{len(val_dataset)}")

# 检查一下掩码的类别数
sample_img, sample_mask = train_dataset[0]
print(f"样本图像形状：{sample_img.shape}")
print(f"样本掩码形状：{sample_mask.shape}")
print(f"掩码像素值范围：{sample_mask.min()} ~ {sample_mask.max()}")  # 应该是 0~36

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# -------------------- 3. 训练配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

num_classes = 37  # 背景(0) + 36种宠物(1~36)
model = UNet(n_channels=3, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 多类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20  # 多类需要更多迭代
best_val_loss = float('inf')


# -------------------- 4. 训练循环（增加IoU计算） --------------------
def compute_iou(pred, target, num_classes):
    """计算每个类别的IoU和平均IoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # 如果该类不存在，忽略
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # 忽略nan类

for epoch in range(1, num_epochs + 1):
    # 训练
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')
    for images, masks in train_loop:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loop.set_postfix({'loss': loss.item()})
    avg_train_loss = train_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_iou += compute_iou(preds, masks, num_classes)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)

    print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val mIoU={avg_val_iou:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_unet_oxford_multiclass.pth')
        print('  -> 保存最佳模型')

print("训练完成！")


# -------------------- 5. 可视化部分验证结果（用彩色显示多类） --------------------
def visualize_predictions(model, loader, device, num_images=3):
    model.eval()
    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images*5))
    # 定义一个固定颜色映射（37类+背景）
    cmap = plt.cm.get_cmap('tab20', num_classes)  # tab20最多20种颜色，但我们可以用jet
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_images:
                break
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            img = images[0].cpu().permute(1,2,0).numpy()
            mask = masks[0].cpu().numpy()
            pred = preds[0]

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('输入图像')
            axes[i, 1].imshow(mask, cmap='jet', vmin=0, vmax=num_classes-1)
            axes[i, 1].set_title('真实掩码')
            axes[i, 2].imshow(pred, cmap='jet', vmin=0, vmax=num_classes-1)
            axes[i, 2].set_title('预测掩码')
            for ax in axes[i]:
                ax.axis('off')
    plt.tight_layout()
    plt.savefig('oxford_pets_multiclass_prediction.png')
    plt.show()

visualize_predictions(model, val_loader, device, num_images=3)
print("可视化结果已保存为 oxford_pets_multiclass_prediction.png")