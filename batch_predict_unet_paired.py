# -*- coding: utf-8 -*-
"""
U-Net 批量预测脚本（Oxford Pets 多类分割）
输入：test_imgs 文件夹中的图片
输出：test_results_paired 文件夹，每张图片生成一个拼合图（原图 + 彩色掩码）
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 1. 定义模型结构（必须与训练时完全一致）--------------------
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


# -------------------- 2. 加载模型和配置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 37  # Oxford Pets 数据集：背景 + 36种宠物
model = UNet(n_channels=3, n_classes=num_classes).to(device)

# 加载训练好的权重（请确认路径正确）
model_path = 'best_unet_oxford_multiclass.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练或修改路径。")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"模型加载成功，使用设备：{device}")

# 预处理变换（与训练时一致：Resize + ToTensor）
image_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


# -------------------- 3. 准备颜色映射 --------------------
# 使用 jet 颜色映射，离散为 num_classes 种颜色
cmap = plt.cm.get_cmap('jet', num_classes)

def mask_to_colored(mask_np):
    """将整数掩码数组 (H,W) 转换为彩色 RGB 图像 (H,W,3) 0-255"""
    # 归一化到 [0,1] 并映射颜色
    colored = cmap(mask_np / (num_classes - 1))[:, :, :3]  # 取 RGB 通道
    colored = (colored * 255).astype(np.uint8)
    return colored

# -------------------- 4. 批量预测并生成拼合图 --------------------
input_folder = 'test_imgs'
output_folder = 'test_results_paired'
os.makedirs(output_folder, exist_ok=True)

valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始处理...\n")

for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    try:
        # 读取原图
        img_pil = Image.open(img_path).convert('RGB')
        # 预处理（缩放到模型输入尺寸）
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)                # (1, 37, H, W)
            pred_mask = torch.argmax(output, dim=1)     # (1, H, W)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()  # (H, W)

        # 生成彩色掩码
        colored_mask = mask_to_colored(pred_mask)

        # 将原图也缩放到相同尺寸用于拼合
        img_resized = img_pil.resize(image_size, Image.Resampling.LANCZOS)
        img_resized_np = np.array(img_resized)  # (H,W,3)

        # 左右拼合
        paired = np.hstack((img_resized_np, colored_mask))
        paired_img = Image.fromarray(paired)

        # 保存结果
        base_name = os.path.splitext(img_file)[0]
        save_path = os.path.join(output_folder, f"{base_name}_paired.png")
        paired_img.save(save_path)

        print(f"✓ {img_file} -> 拼合图已保存至 {save_path}")

    except Exception as e:
        print(f"❌ {img_file} 处理失败: {e}\n")

print(f"\n所有处理完成！结果保存在 {output_folder} 文件夹中。")